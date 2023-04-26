
import wandb


api = wandb.Api()
num_trials = 3
all_missing_runs = 0

projects = {"FFM_ablate": 14.5e6}
for proj, min_timesteps in projects.items():
    print(f"\n\n### {proj} ###")
    # Project is specified by <entity/project-name>
    runs = api.runs(proj)
    names = [r.name for r in runs]
    crashed = [r.state for r in runs]
    num_timesteps = [r.summary.get("timesteps_total", 0) for r in runs]
    envs = [r.config["env"] for r in runs]
    ids = [r.id for r in runs]
    # check for name mismatch
    mismatches = []
    for n, e, i in zip(names, envs, ids):
        env_name = n.split("-")[0]
        true_env = e.split("-")
        if len(true_env) == 1:
            true_env = true_env[0]
        else:
            true_env = true_env[1]

        if true_env != env_name:
            mismatches.append((env_name, true_env, i))

    mismatches = sorted(mismatches)
    if len(mismatches) > 0:
        print(f"{len(mismatches)} mismatches:")
        for n, e, i in mismatches:
            print(f"{n} {e} {i}")
    unique = set(names)
    counts = [names.count(u) for u in unique]
    run_count = {n: c for n, c in dict(zip(unique, counts)).items()}
    crashed_runs = {n: 0 for n in unique}
    premature_runs = {n: 0 for n in unique}
    for name, state, timesteps in zip(names, crashed, num_timesteps):
        #if state == "crashed":
        #    crashed_runs[name] += 1
        if timesteps < min_timesteps:
            premature_runs[name] += 1

    keys = set(run_count.keys()) | set(crashed_runs.keys()) | set(premature_runs.keys())
    missing = {}
    for key in keys:
        total_valid = (
            run_count.get(key, 0)
            - crashed_runs.get(key, 0)
            - premature_runs.get(key, 0)
        )
        total_missing = num_trials - total_valid
        if total_missing > 0:
            missing[key] = total_missing

    # missing = dict(sorted(missing.items(), key=lambda item: item[1]))

    print(f"{proj} missing {sum(missing.values())} runs crashed/early termination:")
    all_missing_runs += sum(missing.values())
    if len(missing) == 0:
        continue
    print(missing)
    # To config
    envs, models = list(zip(*[m.split("-") for m in missing]))
    counts = list(missing.values())

    pretty = [(a, b, c) for a, b, c in sorted(zip(models, envs, counts))]
    models, envs, counts = zip(*pretty)
    pretty = [(a, b, c) for a, b, c in sorted(zip(envs, models, counts))]
    envs, models, counts = zip(*pretty)
    pretty = [(a, b, c) for a, b, c in sorted(zip(counts, envs, models), reverse=True)]
    for p in pretty:
        print(*p)

    # Unique set
    umodels, uenvs, ucounts = list(set(models)), list(set(envs)), min(counts)
    print("Run set")
    umod_str = str(umodels).replace("'", "")
    print(f"models = {umod_str}")
    print(f"env_names = {uenvs}")
    print(f"trials = {ucounts}")
    print("total runs:", len(umodels) * len(uenvs) * ucounts)

    model_types = {
        "S4D": "Convolution",
        "FastWeightProgrammer": "Attention",
        "LinearAttention": "Attention",
        "Frameconv": "Convolution",
        "Framestack": "Convolution",
        "MLP": "Simple",
        "BasicMLP": "Simple",
        "IndRNN": "RNN",
        "LMU": "Convolution",
        "Elman": "RNN",
        "GRU": "RNN",
        "LSTM": "RNN",
    }

print(f"Missing {all_missing_runs} overall")
