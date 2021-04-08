"""Defines the Config object used throughout the project"""
import os
from os.path import join, dirname, splitext
from typing import Dict, Any
from cac.utils.io import read_yml, save_yml

# project home mounted inside docker
HOME = "/workspace/cough-against-covid/"

# user-dependent output directory mounted inside docker
OUT_DIR = "/output/"

# output directory for all users mounted inside docker
ALL_OUT_DIR = '/all-output/'

# directory where the cough data resides (mounted in Docker container)
DATA_ROOT = "/data"


class Config:
    """Class that loads hyperparameters from a yml file.

    :param version: path of the .yml file which contains the hyperparameters,
        the config file should be stored inside `coughagainstcovid/configs/` folder.
        You only need to pass the subpath from this folder.
    :type version: str
    :param user: user whose output folder contains saved config
    :type user: str, defaults to None
    """
    def __init__(self, version: str, user: str = None):
        assert version.endswith('.yml')
        self.version = version
        self.user = user

        if user is not None:
            config_path = join(
                ALL_OUT_DIR, user, version.replace('.yml', ''),
                'config.yml')
            assert os.path.exists(
                config_path), 'Config file does not exist at {}'.format(
                    config_path)

            self.update_from_path(config_path)
            self.modify_loaded_config(user)

        else:
            self.paths = {"HOME": HOME, "OUT_DIR": OUT_DIR}

            config_path = os.path.join(
                self.paths["HOME"],
                "configs",
                version)
            config_subpath = version.replace('.yml', '')
            self.config_save_path = os.path.join(
                self.paths["OUT_DIR"],
                config_subpath,
                "config.yml",
            )
            os.makedirs(dirname(self.config_save_path), exist_ok=True)

            self.output_dir = os.path.join(
                self.paths["OUT_DIR"], config_subpath
            )
            self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
            self.log_dir = os.path.join(self.output_dir, "logs")

            self.update_from_path(config_path)

            # save the config
            self.save()

            # create missing directories
            for path in [self.checkpoint_dir, self.log_dir]:
                os.makedirs(path, exist_ok=True)

    def __repr__(self):
        return "Config(version={})".format(self.version)

    def save(self):
        """Saves parameters"""
        save_yml(self.__dict__, self.config_save_path)

    def update_from_path(self, path: str):
        """Loads parameters from yml file"""
        params = read_yml(path)
        self.update_from_params(params)

    @staticmethod
    def _set_defaults(params: Dict):
        """Validates parameter values"""
        params['metrics_to_track'] = ['precision', 'specificity', 'recall']
        params['allow_val_change'] = params.get('allow_val_change', False)
        params['data']['sampler'] = params['data'].get('sampler', {})
        params['data']['dataset']['params'] = params['data']['dataset'].get(
            'params', {})
        params['model']['subset_tracker'] = params['model'].get(
            'subset_tracker', {})
        # import ipdb; ipdb.set_trace()
        params['model']['subset_tracker']['train'] = params['model']['subset_tracker'].get(
            'train', {})
        params['model']['subset_tracker']['val'] = params['model']['subset_tracker'].get(
            'val', {})

        if 'load' in params['model']:
            load_config = params['model']['load']
            load_config['resume_optimizer'] = load_config.get(
                'resume_optimizer', False)
            load_config['resume_epoch'] = load_config.get(
                'resume_epoch', load_config['resume_optimizer'])
            params['model']['load'] = load_config

        params['model']['eval'] = params['model'].get('eval', {})
        params['model']['eval']['maximize_metric'] = params['model']['eval'].get(
            'maximize_metric', 'specificity')

        # set defaults for aggragation of predictions
        params['model']['eval']['aggregate'] = params['model']['eval'].get('aggregate', {})
        aggregate_config = params['model']['eval']['aggregate']
        aggregate_config['train'] = aggregate_config.get('train', {})
        aggregate_config['val'] = aggregate_config.get('val', {})
        params['model']['eval']['aggregate'] = aggregate_config

        # loss config
        cross_entropy_config = {
            'name': 'cross-entropy',
            'params': {
                'reduction': 'none'
            }
        }
        params['model']['loss'] = params['model'].get('loss', {
            'train': cross_entropy_config,
            'val': cross_entropy_config,
            'all': cross_entropy_config
        })

        return params

    @staticmethod
    def _check_params(params: Dict):
        """Validates parameter values"""
        assert "description" in params
        assert "data" in params
        assert "model" in params

        if 'optimizer' in params['model'] and 'scheduler' in params['model']['optimizer']:
            scheduler_config = params['model']['optimizer']['scheduler']

            if scheduler_config['name'] == 'StepLR':
                assert 'value' not in scheduler_config
            if scheduler_config['name'] == 'MultiStepLR':
                assert 'value' not in scheduler_config
            elif scheduler_config['name'] == 'ReduceLRInPlateau':
                assert scheduler_config['update'] == 'epoch'
                assert 'value' in scheduler_config
            elif scheduler_config['name'] == 'CyclicLR':
                assert scheduler_config['update'] == 'batch'
                assert 'value' not in scheduler_config
            elif scheduler_config['name'] == '1cycle':
                assert scheduler_config['update'] == 'batch'
                assert 'value' not in scheduler_config

        # check loss weight values
        if 'loss_weights' in params['model']:
            adv_loss_weight = params['model']['loss_weights'].get('adv', None)

            # adversarial loss weight should be >= 0
            if adv_loss_weight is not None:
                assert adv_loss_weight['value'] >= 0

        # check multi-task config
        if 'tasks' in params['model']:
            network_config = params['network']['params']['config']
            backbone = network_config['backbone']
            heads = network_config['heads']
            tasks = params['model']['tasks']

            final_dim = None
            for index in reversed(range(len(backbone))):
                layer = backbone[index]
                for _, args in layer.items():
                    if 'out_features' in args:
                        final_dim = args['out_features']
                        break

                if final_dim is not None:
                    break

            for index in range(len(tasks)):
                head = heads[index]
                feed_target = head.get('feed_target', False)
                if feed_target:
                    assert 'target_index' in head

                # if using the output of a head as input for this head
                # that output should have been computed earlier
                if head.get('use_head', False):
                    assert 'head_index' in head
                    assert head['head_index'] < index

        # check aggregation method for combining predictions
        aggregate_config = params['model']['eval']['aggregate']
        if 'at' in aggregate_config['val']:
            assert aggregate_config['val']['at'] in ['softmax','sigmoid']

    def update_from_params(self, params: Dict):
        """Updates parameters from dict"""
        params = self._set_defaults(params)
        self._check_params(params)
        self.__dict__.update(params)

    def modify_loaded_config(self, user: str):
        """Modifies the loaded config to align with user-dependent paths

        :param user: user whose output folder contains saved config
        :type user: str
        """
        USER_OUT_DIR = join(ALL_OUT_DIR, user) + '/'

        PATHS_TO_CHANGE = ['config_save_path', 'output_dir', 'log_dir', 'checkpoint_dir']
        for key in PATHS_TO_CHANGE:
            setattr(self, key, getattr(self, key).replace(OUT_DIR, USER_OUT_DIR))

        PATH_KEYS = ['OUT_DIR']
        for key in PATH_KEYS:
            self.paths[key] = self.paths[key].replace(OUT_DIR, USER_OUT_DIR)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `config.dict['lr']`"""
        return self.__dict__
