import os
import json
import numpy as np
import flask
import ghhops_server as hs
from shapely.geometry import shape, Point
from rasterio.transform import from_origin
from rasterio.features import rasterize
import rasterio
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from PIL import Image

# Load CNN model
cnn_model = tf.keras.models.load_model("cnn_svf_model_new.h5", compile=False)

# Load GNN model
class GATRegressor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False)
        self.linear = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = self.linear(x)
        return x.squeeze(-1)

gnn_model = GATRegressor(in_channels=15, hidden_channels=64)
gnn_model.load_state_dict(torch.load("gnn_tmrt_model_new.pth", map_location="cpu"))
gnn_model.eval()

# Flask + Hops Setup
app = flask.Flask(__name__)
hops = hs.Hops(app)

@hops.component(
    "/full_pipeline",
    name="Full SVF + Tmrt Pipeline",
    description="Generates DSM, CDSM, Building Mask, SVF, and Tmrt from GeoJSON",
    inputs=[
        hs.HopsString("Footprints", "Footprints", "Building footprints GeoJSON", access=hs.HopsParamAccess.ITEM),
        hs.HopsString("Trees", "Trees", "Tree GeoJSON with height and radius", access=hs.HopsParamAccess.ITEM),
        hs.HopsString("Extent", "Extent", "GeoJSON defining the bounds", access=hs.HopsParamAccess.ITEM),
        hs.HopsNumber("PixelSize", "PixelSize", "Pixel size (in meters)", access=hs.HopsParamAccess.ITEM),
        hs.HopsString("OutPath", "PathFolder", "Folder to save all output files", access=hs.HopsParamAccess.ITEM),
        hs.HopsString("Green","Green","Green Area", access=hs.HopsParamAccess.ITEM),
        hs.HopsString("Pavement","Pavement","Pavement Area", access=hs.HopsParamAccess.ITEM),
        hs.HopsNumber("Matrix", "Shadows", "Flattened 128x128 matrix", access=hs.HopsParamAccess.LIST),
    ],
    outputs=[
        hs.HopsString("Status", "Status", "Success or failure message"),
        hs.HopsString("TmrtPNGPath", "Tmrt PNG", "Path to predicted_tmrt.png"),
        hs.HopsString("TmrtMatrix", "Tmrt Matrix Shadow", "2D JSON array of Tmrt values")
    ]
)
def full_pipeline(footprints_str, trees_str, extent_str, pixel_size, out_path, green, pavement, matrix):
    try:
        extent = json.loads(extent_str)
        extent_geom = shape(extent["geometry"] if "geometry" in extent else extent["features"][0]["geometry"])
        minx, miny, maxx, maxy = extent_geom.bounds
        cols = rows = 128
        transform = from_origin(minx, maxy, pixel_size, pixel_size)

        footprints = json.loads(footprints_str)
        building_shapes = [
            (shape(f["geometry"]), float(f.get("properties", {}).get("height", 0)))
            for f in footprints.get("features", [])
        ]

        trees = json.loads(trees_str)
        tree_shapes = []
        for f in trees.get("features", []):
            props = f.get("properties", {})
            height = float(props.get("height", 5))
            radius = float(props.get("radius", 1.5))
            geom = shape(f["geometry"])
            if isinstance(geom, Point):
                tree_shapes.append((geom.buffer(radius), height))

        dsm = rasterize(building_shapes, out_shape=(rows, cols), transform=transform, fill=0, dtype='float32')
        cdsm = rasterize(tree_shapes, out_shape=(rows, cols), transform=transform, fill=0, dtype='float32')
        building_mask = rasterize([s[0] for s in building_shapes], out_shape=(rows, cols), transform=transform,
                                   fill=1, default_value=0, dtype='uint8')

        os.makedirs(out_path, exist_ok=True)

        green_areas = json.loads(green)
        pavement_areas = json.loads(pavement)

        green_shapes = [shape(f["geometry"]) for f in green_areas.get("features", [])]
        pavement_shapes = [shape(f["geometry"]) for f in pavement_areas.get("features", [])]

        landuse = np.zeros((rows, cols), dtype=np.uint8)

        if pavement_shapes:
            pavement_mask = rasterize([(g, 1) for g in pavement_shapes], out_shape=(rows, cols), transform=transform,
                                      fill=0, dtype='uint8')
            landuse[pavement_mask == 1] = 1

        if green_shapes:
            green_mask = rasterize([(g, 5) for g in green_shapes], out_shape=(rows, cols), transform=transform,
                                   fill=0, dtype='uint8')
            landuse[green_mask == 5] = 5

        if building_shapes:
            building_geom = [g[0] for g in building_shapes]
            building_mask = rasterize([(g, 2) for g in building_geom], out_shape=(rows, cols), transform=transform,
                                      fill=0, dtype='uint8')
            landuse[building_mask == 2] = 2

        if matrix and len(matrix) == 128 * 128:
            try:
                custom_array = np.array(matrix).reshape((128, 128))
                custom_array = np.rot90(custom_array, 2)
                shadows = custom_array

                with rasterio.open(os.path.join(out_path, "shadow_calc.tif"), 'w', driver='GTiff', height=128, width=128,
                                   count=1, dtype='float32', crs='EPSG:25831', transform=transform) as dst:
                    dst.write(custom_array, 1)

                cmap_func = plt.get_cmap("Spectral_r")
                norm = np.clip((custom_array - 0) / (1 - 0), 0, 1)
                rgba = cmap_func(norm)[..., :3]
                img = (rgba * 255).astype(np.uint8)
                Image.fromarray(img).save(os.path.join(out_path, "shadow.png"))

            except Exception as matrix_e:
                print(f"Error saving shadow GeoTIFF or PNG: {matrix_e}")

        def save_raster(path, array, dtype):
            with rasterio.open(path, 'w', driver='GTiff', height=rows, width=cols, count=1,
                               dtype=dtype, crs='EPSG:25831', transform=transform) as dst:
                dst.write(array, 1)

        def save_png(path, array, building_mask=None, cmap="Spectral_r", vmin=20, vmax=50):
            cmap_func = plt.get_cmap(cmap)
            norm = np.clip((array - vmin) / (vmax - vmin), 0, 1)
            rgba = cmap_func(norm)[..., :3]
            img = (rgba * 255).astype(np.uint8)

            if building_mask is not None:
                mask = (building_mask != 0)
                img[mask] = 255

            Image.fromarray(img).save(path)

        save_raster(os.path.join(out_path, "dsm.tif"), dsm, 'float32')
        save_raster(os.path.join(out_path, "cdsm.tif"), cdsm, 'float32')
        save_raster(os.path.join(out_path, "buildings.tif"), building_mask, 'uint8')
        save_raster(os.path.join(out_path, "combined_landuse.tif"), landuse, 'uint8')

        # Clip DSM/CDSM
        dsm_raw = np.nan_to_num(np.clip(dsm, 0, 50))
        cdsm_raw = np.nan_to_num(np.clip(cdsm, 0, 50))

        # Normalize DSM/CDSM for CNN (as in training)
        dsm_max = np.max(dsm_raw)
        cdsm_max = np.max(cdsm_raw)

        dsm_cnn = dsm_raw / dsm_max if dsm_max > 0 else np.zeros_like(dsm_raw)
        cdsm_cnn = cdsm_raw / cdsm_max if cdsm_max > 0 else np.zeros_like(cdsm_raw)

        # Predict SVF using CNN
        input_stack = np.stack([dsm_cnn, cdsm_cnn], axis=-1)
        input_stack = np.expand_dims(input_stack, axis=0)
        svf_pred = cnn_model.predict(input_stack, verbose=0)[0, ..., 0]
        svf = np.clip(np.nan_to_num(svf_pred), 0, 1)

        save_raster(os.path.join(out_path, "predicted_svf.tif"), svf, 'float32')
        
        save_png(
            os.path.join(out_path, "predicted_svf.png"),
            svf,
            building_mask=building_mask,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        # Compute Contextual Features
        def compute_contextual_features(dsm, cdsm, landuse, patch_size=8):
            h, w = dsm.shape
            contextual_map = np.zeros((h, w, 7), dtype=np.float32)
            for row in range(0, h, patch_size):
                for col in range(0, w, patch_size):
                    row_end = min(row + patch_size, h)
                    col_end = min(col + patch_size, w)
                    block_dsm = dsm[row:row_end, col:col_end]
                    block_cdsm = cdsm[row:row_end, col:col_end]
                    block_lu = landuse[row:row_end, col:col_end]
                    block_area = block_dsm.size
                    bld_mask = block_dsm > 0
                    mean_bld_height = np.mean(block_dsm[bld_mask]) if np.any(bld_mask) else 0.0
                    building_volume = np.sum(block_dsm[bld_mask]) if np.any(bld_mask) else 0.0
                    volume_density = building_volume / block_area
                    tree_mask = block_cdsm > 0
                    tree_density = np.sum(tree_mask) / block_area
                    mean_tree_height = np.mean(block_cdsm[tree_mask]) if np.any(tree_mask) else 0.0
                    pavement_ratio = np.sum(block_lu == 1) / block_area
                    green_ratio = np.sum(block_lu == 5) / block_area
                    building_ratio = np.sum(block_lu == 2) / block_area
                    patch_features = np.array([
                        mean_bld_height,
                        volume_density,
                        tree_density,
                        mean_tree_height,
                        pavement_ratio,
                        green_ratio,
                        building_ratio
                    ], dtype=np.float32)
                    contextual_map[row:row_end, col:col_end, :] = patch_features
            return contextual_map

        svf = np.nan_to_num(svf_pred)
        svf = np.clip(svf, 0, 1)
        buildings = np.clip(np.nan_to_num(building_mask), 0, 1)
        contextual_map = compute_contextual_features(dsm, buildings, cdsm, landuse)
        h, w = dsm.shape
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        x_coord = xx / w
        y_coord = yy / h

        feature_stack = [
            dsm,
            cdsm,
            svf,
            shadows,
            x_coord,
            y_coord,
            buildings,
            landuse,
            contextual_map[..., 0],
            contextual_map[..., 1],
            contextual_map[..., 2],
            contextual_map[..., 3],
            contextual_map[..., 4],
            contextual_map[..., 5],
            contextual_map[..., 6],
        ]

        x = np.stack(feature_stack, axis=-1).reshape(-1, 15).astype(np.float32)

        edge_index = []
        for row in range(h):
            for col in range(w):
                idx = row * w + col
                if col < w - 1:
                    edge_index.append([idx, idx + 1])
                if row < h - 1:
                    edge_index.append([idx, idx + w])
        edge_index = np.array(edge_index).T

        graph = Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long)
        )

        with torch.no_grad():
            pred_tmrt = gnn_model(graph.x, graph.edge_index).cpu().numpy().reshape(h, w)

        tmrt_tif_path = os.path.join(out_path, "predicted_tmrt.tif")
        tmrt_png_path = os.path.join(out_path, "predicted_tmrt.png")
        save_raster(tmrt_tif_path, pred_tmrt, 'float32')

        save_png(
            os.path.join(out_path, "predicted_tmrt.png"),
            pred_tmrt,
            building_mask=building_mask,
            cmap="Spectral_r",
            vmin=20,
            vmax=50
        )

        return f"Saved DSM, CDSM, SVF and Tmrt to {out_path}", tmrt_png_path, json.dumps(pred_tmrt.tolist())

    except Exception as e:
        return f"Error: {str(e)}", "", "[]"

if __name__ == "__main__":
    app.run(debug=True)
