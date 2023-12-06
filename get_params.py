# with open("params.txt", "w") as f:
 # for g in [0.01, 0.02, 0.03, 0.04, 0.05]:
        # for a in [0.1, 0.3, 0.5, 0.7, 0.9]:
            # for p in [0, 0.1, 0.3, 0.5, 0.7, 0.9]:
                # f.write(f"{g} {a} {p}\n")


# with open("params_sem.txt", "w") as f:
    # for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # f.write(f"{a}\n")

# with open("params_no_prob.txt", "w") as f:
    # for g in [0.01, 0.02, 0.03, 0.04, 0.05]:
        # for a in [0.1, 0.3, 0.5, 0.7, 0.9]:
            # f.write(f"{g} {a}\n")

with open("params_only_pu.txt", "w") as f:
    for margin_factor in [0, 0.1, 0.3, 0.5]:
        for max_lr in [0.00001, 0.0001, 0.001, 0.01]:
            for min_lr_factor in [0.1, 0.01, 0.001]:
                for prior in [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]:
                    for bs in [128, 256, 512]:
                        f.write(f"{margin_factor} {max_lr} {min_lr_factor} {prior} {bs}\n")

        
