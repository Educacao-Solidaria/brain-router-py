-- Inicialização do banco do docker-compose.yml. O entrypoint do postgres roda esta pasta só na 1ª
-- partida (volume pg-data vazio), como o superusuário POSTGRES_USER, no banco POSTGRES_DB, e
-- aborta a partida se um comando falhar (psql com ON_ERROR_STOP). As extensões da migração 0001,
-- com o mesmo IF NOT EXISTS: o `alembic upgrade head` depois as encontra prontas.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
