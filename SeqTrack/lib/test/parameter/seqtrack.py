from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.seqtrack.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/seqtrack/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)

    template_cfg = getattr(cfg.DATA, "TEMPLATE", None)
    data_template_number = getattr(template_cfg, "NUMBER", None) if template_cfg is not None else None
    test_template_number = getattr(cfg.TEST, "NUM_TEMPLATES", None)

    tmpl_override = os.environ.get("SEQTRACK_TEST_NUM_TEMPLATES")
    if tmpl_override:
        override_value = tmpl_override.strip()
        if override_value.lower() == "auto":
            if (
                data_template_number is not None
                and test_template_number is not None
                and test_template_number < data_template_number
            ):
                cfg.TEST.NUM_TEMPLATES = data_template_number
        else:
            try:
                override_int = int(override_value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid SEQTRACK_TEST_NUM_TEMPLATES value: {tmpl_override}"
                ) from exc
            if override_int <= 0:
                raise ValueError("SEQTRACK_TEST_NUM_TEMPLATES must be greater than zero if specified")
            cfg.TEST.NUM_TEMPLATES = override_int
    else:
        if (
            data_template_number is not None
            and test_template_number is not None
            and test_template_number < data_template_number
        ):
            cfg.TEST.NUM_TEMPLATES = data_template_number

    params.cfg = cfg
    print("test config: ", cfg)

    params.yaml_name = yaml_name
    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Allow overriding the evaluation epoch / checkpoint path via env variables.
    override_epoch = os.environ.get("CHECKPOINT_EPOCH")
    override_path = os.environ.get("CHECKPOINT_PATH")

    if override_epoch is not None:
        try:
            override_epoch_int = int(override_epoch)
            cfg.TEST.EPOCH = override_epoch_int
        except ValueError:
            raise ValueError(f"Invalid CHECKPOINT_EPOCH value: {override_epoch}")

    if override_path:
        checkpoint_path = override_path
    else:
        checkpoint_path = os.path.join(
            save_dir,
            "checkpoints/train/seqtrack/%s/SEQTRACK_ep%04d.pth.tar" % (yaml_name, cfg.TEST.EPOCH),
        )

    params.checkpoint = checkpoint_path

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
