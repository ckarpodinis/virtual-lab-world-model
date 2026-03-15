# virtual-lab-world-model

## Rule Extraction Pipeline

Extracts precondition rules from virtual lab instrument simulations.

## Pipeline

### 1. Generate MDP template

```bash
python lab_inventory.py
```

Saves an MDP template for a lab instrument (e.g. electronic scale) to a JSON file:

```
scale_mdp_template.json
```

---

### 2. Generate MDPs

```bash
python mdp_generator.py scale_mdp_template.json -n 100 -o scale_mdps.jsonl
```

| Argument | Description |
|---|---|
| `scale_mdp_template.json` | Input MDP template |
| `-n 100` | Number of MDPs to generate |
| `-o scale_mdps.jsonl` | Output file |

---

### 3. Build world model

```bash
python build_world_model.py scale_mdps.jsonl -o scale_world_model.json
```

| Argument | Description |
|---|---|
| `scale_mdps.jsonl` | Input MDPs |
| `-o scale_world_model.json` | Output world model |

---

### 4. Extract rules

```bash
python extract_rules.py scale_world_model.json --threshold 0.7
```

| Argument | Description |
|---|---|
| `scale_world_model.json` | Input world model |
| `--threshold 0.7` | Reward threshold — actions with expected reward `>= threshold` are considered valid (default: `0.7`) |

