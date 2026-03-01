# bash scripts/train_policy_multi.sh  simple_dp3  adroit_hammer  baseline_compare  7
# bash scripts/train_policy_multi.sh  simple_dp3  adroit_door  baseline_compare  7   #最后跑吧，跑了一半停掉了，差最后一个种子
#   data/outputs/adroit_door-simple_dp3-baseline_compare
# bash scripts/train_policy.sh  dit-wm  adroit_pen  baseline_compare_no_fusion_no_wm  0 7
# bash scripts/train_policy.sh  dit-wm  adroit_door  baseline_compare_no_fusion_no_wm  0 7
# bash scripts/train_policy_multi.sh  dit-wm  adroit_hammer  baseline_compare  7
bash scripts/train_policy_multi.sh  dit-wm  adroit_pen  baseline_compare_5ep_ablation  3
bash scripts/train_policy_multi.sh  dit-wm  adroit_door  baseline_compare_5ep_ablation  3   
bash scripts/train_policy_multi.sh  dit-wm  metaworld_reach-wall  baseline_compare_5ep_ablation  3
