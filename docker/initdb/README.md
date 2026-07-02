# Postgres init scripts

Any `*.sql` or `*.sh` file in this directory is executed **once**, in
alphabetical order, the first time Postgres boots against an empty `pgdata`
volume (via the official image's `/docker-entrypoint-initdb.d` hook).

Use it to create the schema the loader expects (`glucose_entries`, `users`,
`user_settings`, …). To re-run after editing, reset the volume:

    docker compose down -v && docker compose up -d

This runs only on a fresh volume — it will not migrate an existing database.
