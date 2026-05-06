PROBLEM=$1
OUTPUT=$2

/home/stu1/zhaozhongxu/prove/eprover/PROVER/eprover \
	-sRp \
	--auto-schedule \
	--cpu-limit=11 \
	--soft-cpu-limit=10 \
	--free-numbers \
	--memory-limit=10000 \
	--print-statistics \
	--tstp-format \
	$PROBLEM > $OUTPUT 2> $OUTPUT.err
