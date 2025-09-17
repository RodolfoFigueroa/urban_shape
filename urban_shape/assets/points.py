import dagster as dg
import geopandas as gpd
from hdbscan import HDBSCAN

from urban_shape.partitions import zone_partitions
from urban_shape.resources import ClusterConfigResource


def points_factory(top_prefix: str) -> dg.AssetsDefinition:
    @dg.asset(
        name="base",
        key_prefix=[top_prefix, "points"],
        ins={"df_points": dg.AssetIn(["denue", "split"])},
        partitions_def=zone_partitions,
        required_resource_keys=set([f"{top_prefix}_config_resource"]),
        io_manager_key="geodataframe_manager",
        group_name=f"points_{top_prefix}",
    )
    def _asset(
        context: dg.AssetExecutionContext,
        df_points: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        config: ClusterConfigResource = getattr(
            context.resources,
            f"{top_prefix}_config_resource",
        )

        crs = df_points.estimate_utm_crs()

        X = df_points.to_crs(crs)["geometry"].get_coordinates().to_numpy() / 100

        model = HDBSCAN(
            min_samples=config.min_samples,
            min_cluster_size=config.min_cluster_size,
            cluster_selection_epsilon=config.eps,
            core_dist_n_jobs=1,
        )
        model.fit(X)

        df_points["label"] = model.labels_
        return df_points

    return _asset


def points_remove_unused_factory(top_prefix: str) -> dg.AssetsDefinition:
    @dg.asset(
        name="remove_unused",
        key_prefix=[top_prefix, "points"],
        ins={"df_points": dg.AssetIn([top_prefix, "points", "base"])},
        partitions_def=zone_partitions,
        io_manager_key="geodataframe_manager",
        group_name=f"points_{top_prefix}",
    )
    def _asset(df_points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        df_points = df_points.query("label != -1")
        label_counts = df_points["label"].value_counts()
        small_labels = label_counts[label_counts < 10].index.tolist()
        return df_points[~df_points["label"].isin(small_labels)]

    return _asset


dassets = [
    factory(top_prefix)
    for factory in (points_factory, points_remove_unused_factory)
    for top_prefix in ("coarse", "fine")
]
