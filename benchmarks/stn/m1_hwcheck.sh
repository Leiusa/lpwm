# Hardware gate for every milestone-1 job. Source this file, then call `m1_check_gpu`.
# Returns 0 only if exactly one GPU is visible, its nvidia-smi name is exactly "NVIDIA RTX A6000", and the node is not
# grogu-4-13. (The Ada variant is named "NVIDIA RTX 6000 Ada Generation" and a 3080 Ti "NVIDIA GeForce RTX 3080 Ti":
# both fail.) M1_TEST_HOST / M1_TEST_SMI let the logic be unit-tested without a GPU.
m1_check_gpu() {
  local host="${M1_TEST_HOST:-$(hostname -s)}"
  local exp="${M1_EXPECTED_GPU:-NVIDIA RTX A6000}"
  local smi
  if [ -n "${M1_TEST_SMI:-}" ]; then smi="$M1_TEST_SMI"; else smi="$(nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader,nounits 2>&1)"; fi
  local n; n=$(printf '%s\n' "$smi" | sed '/^$/d' | wc -l | tr -d ' ')
  if [ "$host" = "grogu-4-13" ]; then echo "HW_MISMATCH node $host is excluded"; return 3; fi
  if [ "$n" != "1" ]; then echo "HW_MISMATCH expected exactly one visible GPU, saw $n: $smi"; return 3; fi
  case "$smi" in
    "$exp, GPU-"*) ;;
    *) echo "HW_MISMATCH GPU line is not '$exp': $smi"; return 3;;
  esac
  echo "HW_OK host=$host gpu=$smi"
  return 0
}
