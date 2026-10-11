#!/usr/bin/env bash
# PostgreSQL 16 + pgvector on a cluster WITHOUT Docker, via Apptainer. UNTESTED here (the build sandbox has no Apptainer
# and cannot reach Docker registries). All state lives under $PGROOT (default: next to this repo on /blue), never in $HOME.
#
#   bash scripts/hipergator_postgres.sh start     # first run pulls the image (needs internet) and initialises the cluster
#   bash scripts/hipergator_postgres.sh status
#   bash scripts/hipergator_postgres.sh stop
# Then:  export DATABASE_URL=postgresql://rag:rag@localhost:5432/ragtest   (this is already the code's default)
# Run it on the SAME node as the ingest / API / eval commands (localhost). If the port is taken, set PGPORT.
set -euo pipefail
EXT=${EXT:?source external/_env/env.sh first (sets EXT and the cache locations)}
PGROOT=${PGROOT:-$EXT/_pg}
PGPORT=${PGPORT:-5432}
IMG=$PGROOT/pgvector_pg16.sif
DATA=$PGROOT/data
RUN=$PGROOT/run
mkdir -p "$PGROOT" "$RUN"
PGBIN=/usr/lib/postgresql/16/bin           # location inside the pgvector/pgvector:pg16 (Debian) image
ap() { apptainer exec --bind "$PGROOT:$PGROOT" "$IMG" "$@"; }

case "${1:-}" in
  start)
    [[ -f $IMG ]] || (cd "$PGROOT" && apptainer pull pgvector_pg16.sif docker://pgvector/pgvector:pg16)
    if [[ ! -d $DATA ]]; then
      ap $PGBIN/initdb -D "$DATA" -U rag --auth=trust
    fi
    ap $PGBIN/pg_ctl -D "$DATA" -o "-p $PGPORT -k $RUN -c listen_addresses=127.0.0.1" -l "$PGROOT/postgres.log" -w start
    ap $PGBIN/psql -h 127.0.0.1 -p "$PGPORT" -U rag -d postgres -tc "SELECT 1 FROM pg_database WHERE datname='ragtest'" | grep -q 1 \
      || ap $PGBIN/createdb -h 127.0.0.1 -p "$PGPORT" -U rag ragtest
    echo "ready: postgresql://rag:rag@localhost:$PGPORT/ragtest  (log: $PGROOT/postgres.log)"
    ;;
  stop)   ap $PGBIN/pg_ctl -D "$DATA" -m fast stop ;;
  status) ap $PGBIN/pg_ctl -D "$DATA" status ;;
  *) echo "usage: $0 start|stop|status"; exit 2 ;;
esac
