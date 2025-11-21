#!/bin/bash

##########################################################################
### MASTER SCRIPT: Run All Tier 1 Experiments                         ###
##########################################################################
# This script submits all 4 Tier 1 experiments with dependencies so they
# run sequentially (one at a time). This prevents GPU conflicts and makes
# it easier to monitor progress.
#
# Total estimated time: 5-6 days (including queue time)
# Total GPU-hours: 384 (96 per experiment × 4 experiments)
##########################################################################

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "Submitting Tier 1 Experimental Suite"
echo "=========================================="

# EXP 1A: Conservative weighted (HIGHEST PRIORITY)
JOB1=$(sbatch --parsable run_tier1_exp1a.sh)
echo "✓ EXP 1A submitted: Job ID ${JOB1} (α=0.1, most likely to succeed)"

# EXP 1B: Moderate weighted (depends on 1A)
JOB2=$(sbatch --parsable --dependency=afterany:${JOB1} run_tier1_exp1b.sh)
echo "✓ EXP 1B submitted: Job ID ${JOB2} (α=0.25, after 1A)"

# EXP 1C: Longer training (depends on 1B)
JOB3=$(sbatch --parsable --dependency=afterany:${JOB2} run_tier1_exp1c.sh)
echo "✓ EXP 1C submitted: Job ID ${JOB3} (60 epochs, after 1B)"

# EXP 1D: Control (depends on 1C)
JOB4=$(sbatch --parsable --dependency=afterany:${JOB3} run_tier1_exp1d.sh)
echo "✓ EXP 1D submitted: Job ID ${JOB4} (MAE control, after 1C)"

echo "=========================================="
echo "All 4 experiments queued!"
echo ""
echo "Job chain:"
echo "  ${JOB1} (1A) → ${JOB2} (1B) → ${JOB3} (1C) → ${JOB4} (1D)"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/mae_grad_*"
echo ""
echo "Expected completion: 5-6 days from now"
echo "=========================================="

# Create a status file
cat > tier1_job_status.txt << EOF
Tier 1 Experimental Suite - Job IDs
====================================
EXP 1A (α=0.1):       ${JOB1}
EXP 1B (α=0.25):      ${JOB2}  
EXP 1C (60 epochs):   ${JOB3}
EXP 1D (control):     ${JOB4}

Submission time: $(date)

Monitor: squeue -u \$USER
Cancel: scancel ${JOB1} ${JOB2} ${JOB3} ${JOB4}
EOF

echo "Job IDs saved to: tier1_job_status.txt"
