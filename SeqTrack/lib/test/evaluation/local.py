from pathlib import Path

from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    # Base directory for generated artifacts (testing, result plots, etc.)
    repo_root = Path(__file__).resolve().parents[4]
    save_dir = repo_root

    settings.prj_dir = str(repo_root / "SeqTrack")
    settings.save_dir = str(save_dir)
    settings.results_path = str(save_dir / "testing" / "results")
    settings.segmentation_path = str(save_dir / "testing" / "segmentation")
    settings.network_path = str(save_dir / "testing" / "networks")
    settings.result_plot_path = str(save_dir / "testing" / "result_plots")

    # Dataset roots (left empty unless explicitly available locally)
    settings.davis_dir = ''
    settings.got10k_lmdb_path = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = ''
    settings.lasot_lmdb_path = ''
    settings.lasot_path = ''
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.tc128_path = ''
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

