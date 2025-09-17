import dagster as dg


class PathResource(dg.ConfigurableResource):
    jobs_path: str
    data_path: str
    population_grids_path: str
    scian_path: str


class ClusterConfigResource(dg.ConfigurableResource):
    eps: int
    min_samples: int
    min_cluster_size: int


class MedoidConfigResource(dg.ConfigurableResource):
    n_clusters: int
