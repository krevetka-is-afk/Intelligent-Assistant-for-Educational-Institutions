#!/usr/bin/env bash
# build-finals.sh — сборка всех документов курсовой в PDF
# Результаты складываются в docs/coursework-docs/finals/
#
# Использование:
#   ./build-finals.sh          # собрать все
#   ./build-finals.sh tz ro    # собрать только выбранные (tz, tz-team, pmi, pmi-team, pz, ro)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TZ_ROOT="$(cd "$SCRIPT_DIR/../technical-specification-for-IAfEI" && pwd)"
FINALS="$SCRIPT_DIR/finals"

# ── цвета ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC}  $*"; }
warn() { echo -e "${YELLOW}!${NC}  $*"; }
fail() { echo -e "${RED}✗${NC}  $*"; }

# ── проверка typst ──────────────────────────────────────────────────────────
if ! command -v typst &>/dev/null; then
    fail "typst не найден. Установите: https://github.com/typst/typst/releases"
    exit 1
fi

mkdir -p "$FINALS"

# ── таблица документов ──────────────────────────────────────────────────────
# Формат: "ключ|путь к main.typ|имя выходного PDF|--root (пусто = каталог main.typ)"
declare -a DOCS=(
    "tz|$TZ_ROOT/ТЗ-инд-растворов/main.typ|ТЗ-Растворов.pdf|"
    "tz-team|$TZ_ROOT/ТЗ-общее/main.typ|ТЗ-общее.pdf|"
    "pmi|$SCRIPT_DIR/docs/pmi/main.typ|ПМИ-Растворов.pdf|$SCRIPT_DIR"
    "pmi-team|$SCRIPT_DIR/docs/pmi-team/main.typ|ПМИ-общее.pdf|$SCRIPT_DIR"
    "pz|$SCRIPT_DIR/docs/pz/main.typ|ПЗ-Растворов.pdf|$SCRIPT_DIR"
    "ro|$SCRIPT_DIR/docs/ro/main.typ|РО-Растворов.pdf|$SCRIPT_DIR"
)

# ── фильтр по аргументам ────────────────────────────────────────────────────
FILTER=("$@")   # пусто → собирать все

should_build() {
    local key="$1"
    [[ ${#FILTER[@]} -eq 0 ]] && return 0
    for f in "${FILTER[@]}"; do
        [[ "$f" == "$key" ]] && return 0
    done
    return 1
}

# ── сборка ─────────────────────────────────────────────────────────────────
BUILT=0
SKIPPED=0
FAILED=0

echo ""
echo "  Сборка документов IAfEI → $FINALS"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  ─────────────────────────────────────────────────────────"

for entry in "${DOCS[@]}"; do
    IFS='|' read -r key src out root <<< "$entry"

    if ! should_build "$key"; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    if [[ ! -f "$src" ]]; then
        fail "$key: исходный файл не найден — $src"
        FAILED=$((FAILED + 1))
        continue
    fi

    # собираем аргументы компиляции
    COMPILE_ARGS=()
    [[ -n "$root" ]] && COMPILE_ARGS+=(--root "$root")
    COMPILE_ARGS+=("$src" "$FINALS/$out")

    printf "  %-12s %s ... " "$key" "$out"
    if typst compile "${COMPILE_ARGS[@]}" 2>/tmp/typst-err-"$key".log; then
        ok "готово"
        BUILT=$((BUILT + 1))
    else
        echo ""
        fail "$key: ошибка компиляции"
        sed 's/^/    /' /tmp/typst-err-"$key".log >&2
        FAILED=$((FAILED + 1))
    fi
done

echo "  ─────────────────────────────────────────────────────────"
echo "  Собрано: $BUILT  |  Пропущено: $SKIPPED  |  Ошибок: $FAILED"
echo ""

[[ $FAILED -eq 0 ]] && exit 0 || exit 1
