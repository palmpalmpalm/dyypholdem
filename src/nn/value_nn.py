
import os
from pathlib import Path

import torch

import settings.arguments as arguments

from nn.compact_value_net import is_compact_checkpoint, load_compact_checkpoint
import nn.modules.module


class ValueNn(object):

    model_info: dict
    model_state: dict
    model: nn.modules.module

    def __init__(self):
        self.model_info = {}
        self.model_state = {}

    def __repr__(self):
        repr_str = "Model info: "
        repr_str = repr_str + f"street={self.model_info['street']}, epoch={self.model_info['epoch']}, validation loss={self.model_info['valid_loss']:0.6f}, "
        repr_str = repr_str + f"device={'cpu' if self.model_info['device'] == torch.device('cpu') else 'cuda'}, "
        repr_str = repr_str + f"datatype={'float32' if self.model_info['datatype'] is torch.float32 else 'float64'}"
        return repr_str

    # --- Gives the neural net output for a batch of inputs.
    # -- @param inputs An NxI tensor containing N instances of neural net inputs.
    # -- See @{net_builder} for details of each input.
    # -- @param output An NxO tensor in which to store N sets of neural net outputs.
    # -- See @{net_builder} for details of each output.
    def get_value(self, inputs, output):
        with torch.inference_mode():
            output.copy_(self.model.forward(inputs))

    def load_for_street(self, street, aux=False, training=False):
        compact_root = os.environ.get("DYYPHOLDEM_COMPACT_MODEL_PATH")
        if compact_root and not training:
            if aux:
                assert street == 1
                folder = "preflop-aux"
            else:
                folder = arguments.street_folders[street + 1].strip("/")
            net_file = Path(compact_root) / folder / "final_compact.pt"
            return self.load_from_file(str(net_file))

        if training:
            net_file = arguments.training_model_path
        else:
            # load final model for specific street
            net_file = arguments.model_path
        if aux:
            assert street == 1
            net_file = net_file + "preflop-aux/"
        else:
            net_file += arguments.street_folders[street + 1]
        net_file = net_file + arguments.value_net_name
        net_file = net_file + ".tar"

        return self.load_from_file(net_file)

    def load_from_file(self, file_name: str):

        arguments.timer.split_start(f"Loading neural network '{file_name}'", log_level="DEBUG")

        saved_dict = self._load_checkpoint(file_name)
        if is_compact_checkpoint(saved_dict):
            self.model = load_compact_checkpoint(saved_dict)
            self.model_state = {}
            self.model_info = dict(saved_dict["model_info"])
            self.model_info["device"] = torch.device(
                str(self.model_info.get("device", "cpu"))
            )
            datatype = self.model_info.get("datatype", "float32")
            if isinstance(datatype, str):
                datatype = getattr(torch, datatype)
            self.model_info["datatype"] = datatype
        else:
            self.__dict__.update(saved_dict)

        assert self.model, "no model found in file"
        assert self.model_info, "no model info found in file"

        arguments.logger.trace(repr(self))
        device = self.model_info['device']
        if device and arguments.device != device:
            if arguments.device == torch.device('cuda'):
                arguments.logger.info("Moving model from CPU to GPU")
                self.model.cuda()
            elif arguments.device == torch.device('cpu'):
                arguments.logger.info("Moving model from GPU to CPU")
                self.model.cpu()
            else:
                raise ValueError("unknown target device")
        elif not device:
            arguments.logger.warning(f"Model does not contain device information - setting it to {repr(arguments.device)}")
            if arguments.device == torch.device('cpu'):
                self.model.cpu()
            else:
                self.model.cuda()

        # The legacy module graph uses ``evaluate`` while native PyTorch uses
        # ``eval``. Compact checkpoints deliberately use the native API.
        if hasattr(self.model, "evaluate"):
            self.model.evaluate()
        else:
            self.model.eval()

        arguments.timer.split_stop("Network loaded in", log_level="LOADING")

        return self

    @staticmethod
    def _load_checkpoint(file_name: str):
        try:
            return torch.load(file_name, map_location="cpu", weights_only=True)
        except TypeError:
            # PyTorch versions before ``weights_only`` was introduced.
            return torch.load(file_name, map_location="cpu")
        except Exception:
            # Legacy checkpoints pickle the custom module graph and therefore
            # cannot be loaded by the restricted weights-only unpickler.
            return torch.load(file_name, map_location="cpu", weights_only=False)

    def load_info_from_file(self, file_name: str):

        arguments.timer.split_start(f"Loading neural network '{file_name}'", log_level="DEBUG")

        saved_dict = torch.load(file_name)
        self.__dict__.update(saved_dict)

        assert self.model, "no model found in file"
        assert self.model_info, "no model info found in file"

        arguments.timer.split_stop("Network loaded in", log_level="LOADING")

        return self

    def save_model(self, model, file_name, state=None, **kwargs):

        for key in kwargs:
            self.model_info[key] = kwargs[key]
        if 'device' not in self.model_info.keys():
            self.model_info['device'] = arguments.device
        if 'datatype' not in self.model_info.keys():
            self.model_info['datatype'] = torch.float32

        self.model = model

        if state is not None:
            self.model_state = state

        arguments.logger.info(f"Saving model '{file_name}'")
        arguments.logger.debug(f"{repr(self)}")
        torch.save({'model_info': self.model_info, 'model': self.model, 'model_state': self.model_state}, file_name)
