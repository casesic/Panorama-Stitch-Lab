import json
import numpy as np
import pytest

from lib_pano.fitting_io.densekp_delivery_io import (
    load_densekp_delivery_observation, load_densekp_delivery_observations,
)


def _person(x):
    return {"point_ids":[0],"pairs":[[x,50,x+1,50]],"valid":[1]}


def test_v1_singular_loader_backward_compatible(tmp_path):
    path=tmp_path/"v1.json"
    path.write_text(json.dumps({"schema_version":"densekp_pairs_v1","status":"matched","coord_space":"erp_rotate",**_person(10)}))
    observation=load_densekp_delivery_observation(path,image_width=100,image_height=100)
    assert observation.valid.tolist()==[True]


def test_v2_plural_loader_and_singular_rejection(tmp_path):
    path=tmp_path/"v2.json"
    path.write_text(json.dumps({"schema_version":"densekp_pairs_v2","status":"matched","coord_space":"erp_rotate","persons":[_person(10),_person(20)]}))
    assert len(load_densekp_delivery_observations(path,image_width=100,image_height=100))==2
    with pytest.raises(ValueError,match="exactly one"):
        load_densekp_delivery_observation(path,image_width=100,image_height=100)
