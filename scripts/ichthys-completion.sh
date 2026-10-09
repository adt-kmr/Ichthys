# Ichthys bash completion
_ichthys_completion() {
    local prev="${COMP_WORDS[COMP_CWORD-1]}"
    local curr="${COMP_WORDS[COMP_CWORD]}"

    case "${prev}" in
        ichthys-train)
            COMPREPLY=($(compgen -W '--checkpoint_dir --epochs --warmup --val_every --selection_metric --debug --no_dino --no_global_attention --dino_repo_dir --dino_weights --greedygroup --linearsumgroup --train_workers --val_workers' -- "${curr}"))
            ;;
        ichthys-infer)
            COMPREPLY=($(compgen -W '--scene_json --images_root --output --thr_assoc --alpha --beta --dist_thresh --sim_thresh --max_gap --min_conf --min_obs --no_rope --no_predictor --workers --fp16 --use_dino' -- "${curr}"))
            ;;
        ichthys-evaluate)
            COMPREPLY=($(compgen -W '--pred --gt --dist_thr --t_min --t_max --gt_t_offset --pred_t_offset --ignore_empty_frames --match_pred_range' -- "${curr}"))
            ;;
        ichthys-sort3d)
            COMPREPLY=($(compgen -W '--input --output --gt --result --dist-thr --gate --max-age --min-hits --process-noise --measurement-noise --no-fill-misses' -- "${curr}"))
            ;;
        ichthys-viz)
            COMPREPLY=($(compgen -W '--workspace' -- "${curr}"))
            ;;
        *)
            COMPREPLY=()
            ;;
    esac
}
complete -F _ichthys_completion ichthys-train ichthys-infer ichthys-evaluate ichthys-sort3d ichthys-viz