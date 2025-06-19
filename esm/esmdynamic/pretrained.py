from pathlib import Path

import torch

from esm.esmdynamic.esmdynamic import ESMDynamic


def _load_model(model_name, model_object=ESMDynamic):
    if model_name.endswith(".pt"):  # local, treat as filepath --> technically not used
        model_path = Path(model_name)
        model_data = torch.load(str(model_path), map_location="cpu")
    else:  # load from data repository, if file exists it won't redownload
        url = "{}" TODO: PUT URL HERE!!!!
        model_data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")

    model = model_object()

    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())

    missing_essential_keys = []
    for missing_key in expected_keys - found_keys:
        if not missing_key.startswith("esm."):
            missing_essential_keys.append(missing_key)

    if missing_essential_keys:
        raise RuntimeError(f"Keys '{', '.join(missing_essential_keys)}' are missing.")

    model.load_state_dict(model_state, strict=False)

    return model


def esmdynamic_v1():
    """
    Load esmdynamic_v1 with pretrained weights.
    """
    return _load_model("esmdynamic_v1")