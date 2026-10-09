# Ichthys zsh completion
_ichthys_completion() {
    local si=$IFS
    IFS=$'\n'
    local commands=(
        'ichthys-train:Train Ichthys on multi-camera detection datasets'
        'ichthys-infer:Run 3D tracking inference'
        'ichthys-evaluate:Evaluate 3D tracking with MOT metrics'
        'ichthys-sort3d:Run SORT3D baseline tracker'
        'ichthys-viz:Run Ichthys dashboard'
    )
    local flags_by_cmd=(
        'ichthys-train:--checkpoint_dir --epochs --warmup --val_every --selection_metric --debug --no_dino --no_global_attention --dino_repo_dir --dino_weights --greedygroup --linearsumgroup --train_workers --val_workers'
        'ichthys-infer:--scene_json --images_root --output --thr_assoc --alpha --beta --dist_thresh --sim_thresh --max_gap --min_conf --min_obs --no_rope --no_predictor --workers --fp16 --use_dino'
        'ichthys-evaluate:--pred --gt --dist_thr --t_min --t_max --gt_t_offset --pred_t_offset --ignore_empty_frames --match_pred_range'
        'ichthys-sort3d:--input --output --gt --result --dist-thr --gate --max-age --min-hits --process-noise --measurement-noise --no-fill-misses'
        'ichthys-viz:--workspace'
    )
    local si=$IFS
    IFS=$'\n'
    cur="${words[CURRENT]}"
    prev="${words[CURRENT-1]}"
    
    if [[ -n "$prev" ]]; then
        for cmd_info in "${commands[@]}"; do
            cmd="${cmd_info%%:*}"
            desc="${cmd_info#*:}"
            if [[ "$prev" == "$cmd" ]]; then
                local flags="${flags_by_cmd[$((++idx))]}"
                COMPREPLY=($(compgen -W "$flags" -- "$cur"))
                return 0
            fi
        done
    fi
    
    IFS=$si
}
EOF
echo "Created zsh completion skeleton"