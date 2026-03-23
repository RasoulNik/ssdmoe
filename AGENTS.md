# AGENTS.md

Use `./scripts/mlx-server.sh` to manage the MLX backend server.

Commands:

```bash
./scripts/mlx-server.sh start
./scripts/mlx-server.sh start 9b
./scripts/mlx-server.sh status
./scripts/mlx-server.sh stop
./scripts/mlx-server.sh restart
./scripts/mlx-server.sh restart 9b
```

Defaults:

- Default model: `mlx-community/Qwen3.5-4B-MLX-4bit`
- Optional preset: `9b` -> `mlx-community/Qwen3.5-9B-MLX-4bit`
- Host: `127.0.0.1`
- Port: `9000`
- Base URL: `http://127.0.0.1:9000/v1`
- Log file: `.run/mlx-lm.log`
- Default max tokens: `16384`
- Thinking: enabled by default

If Docker or another external client needs access, run:

```bash
HOST=0.0.0.0 ./scripts/mlx-server.sh start
```

Use `./scripts/open-webui.sh` to run Open WebUI on top of the MLX server without Docker.

Commands:

```bash
./scripts/open-webui.sh start
./scripts/open-webui.sh status
./scripts/open-webui.sh stop
```

Defaults:

- Open WebUI URL: `http://127.0.0.1:3001`
- MLX backend URL: `http://127.0.0.1:9000/v1`
- Open WebUI log file: `.run/open-webui.log`
