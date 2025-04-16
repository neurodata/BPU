import copy
import os
import yaml
from config import chess_config as config_lib
from DPU_lib.chess_utils import data_loader as data_loader
from DPU_lib.train_utils import training
from DPU_lib.evaluate_on_puzzles.evaluate_DPU_on_puzzles import eval_DPU_on_puzzles
from DPU_lib.train_utils.training import get_out_path

def main():
  with open("config/droso_config_KC.yaml", "r") as f:
      base_config = yaml.safe_load(f)
  batch_size = base_config.pop("batch_size")
  num_epoch = base_config.pop("num_epoch")
  num_records = base_config.pop("train_num_sample")
  experiments = base_config.pop('experiments')
  for base_config['timesteps'] in [8]:
      for learnable_KC in ['between',None, 'within',]:
          for fixed_fc in [False]:# training from scratch now (model incompatable)
              for exp_id in experiments.keys():
                  learnable_name = '_' + learnable_KC + 'KC' if learnable_KC is not None else '_All'
                  droso_config = copy.deepcopy(base_config)
                  droso_config = {**droso_config, **experiments[exp_id]}
                  droso_config['exp_id'] = exp_id
                  droso_config['fixed_fc'] = fixed_fc
                  droso_config['learnable_KC'] = learnable_KC

                  assert droso_config['trainable'] == True
                  if 'filter_num' in droso_config:
                      droso_config['exp_id'] = droso_config['exp_id'] + f"_{droso_config['filter_num']}filters"
                  droso_config['exp_id'] = droso_config['exp_id'] + learnable_name
                  if fixed_fc:
                      droso_config['exp_id'] = droso_config['exp_id'] + 'fixed_fc'
                  droso_config['exp_id'] = droso_config['exp_id'] + f"_{str(num_records)}"

                  out_path = get_out_path(droso_config)
                  if os.path.exists(os.path.join(out_path, "model.pth")):
                      continue

                  model_choice = droso_config["model_choice"]
                  policy = droso_config['policy']

                  # TODO not sure if we'd want to bin the probs, maybe try both -> not using it
                  num_return_buckets = 128 if droso_config['data']['use_bucket'] else None

                  train_config = config_lib.TrainConfig(
                      learning_rate=0.0003, #1e-4,
                      num_epoch = num_epoch,
                      data=config_lib.DataConfig(
                          model_choice=model_choice,
                          batch_size=batch_size,
                          shuffle=True,
                          worker_count=0,  # 0 disables multiprocessing.
                          num_return_buckets=num_return_buckets,
                          policy=policy,
                          split='train',
                          num_records = num_records, # total is 530310443 for SV
                      ),
                  )
                  test_config = config_lib.EvalConfig(
                      data=config_lib.DataConfig(
                          model_choice=model_choice,
                          batch_size=batch_size,
                          shuffle=False,
                          worker_count=0,  # 0 disables multiprocessing.
                          num_return_buckets=num_return_buckets,
                          policy=policy,  # pytype: disable=wrong-arg-types
                          split='test',
                          num_records=num_records//10,
                      ),
                  )

                  model,out_path = training.train(
                      train_config=train_config,
                      test_config = test_config,
                      build_data_loader=data_loader.build_data_loader,
                      droso_config = droso_config,
                  )

                  eval_DPU_on_puzzles(out_path)
                  # eval_DPU_on_puzzles(out_path,score_filter=500)
if __name__ == '__main__':
  main()
