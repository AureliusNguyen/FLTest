#!/bin/bash

# Run all FL frameworks with thorough testing configuration
cd /home/mujtaba/fl-frameworks-testing
export PYTHONPATH=/home/mujtaba/fl-frameworks-testing

RESULTS_FILE="framework_results.txt"

echo "=============================================="
echo "FL Framework Comparison - Thorough Test"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  - Dataset: MNIST (60k training samples)"
echo "  - DATASET_DIVISION_CLIENTS: 10 (6000 samples/client)"
echo "  - num_clients: 10"
echo "  - num_rounds: 10"
echo "  - client_epochs: 1"
echo "  - client_batch_size: 256 (optimized for GPU)"
echo "  - server_batch_size: 1024"
echo "  - device: cuda (GTX 1060 6GB)"
echo ""

# Clear old results
echo "FL Framework Comparison Results - $(date)" > $RESULTS_FILE
echo "==============================================" >> $RESULTS_FILE
echo "Config: DATASET_DIVISION_CLIENTS=10, num_clients=10, num_rounds=10, device=cuda, batch=256/1024" >> $RESULTS_FILE

# Clear cached data to ensure fresh dataset partitioning
rm -rf data/dataset_cache 2>/dev/null
rm -rf data/caches 2>/dev/null
echo "Cleared dataset and result caches for fresh run."
echo ""

for framework in flower flare pfl; do
    echo "----------------------------------------------"
    echo "Running $framework..."
    echo "----------------------------------------------"

    echo "" >> $RESULTS_FILE
    echo "=== $framework ===" >> $RESULTS_FILE

    START_TIME=$(date +%s)

    # Run with thorough config: more data per client + GPU + optimized batch sizes
    OUTPUT=$(poetry run python fl_testing/scripts/main.py \
        framework=$framework \
        DATASET_DIVISION_CLIENTS=10 \
        num_clients=10 \
        num_rounds=10 \
        device=cuda \
        client_batch_size=256 \
        server_batch_size=1024 \
        2>&1)

    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    # Extract only the final results (last 25 lines contain the summary)
    echo "$OUTPUT" | tail -25 >> $RESULTS_FILE
    echo "Duration: ${DURATION}s" >> $RESULTS_FILE
    echo "Exit code: $EXIT_CODE" >> $RESULTS_FILE

    echo "$framework completed in ${DURATION}s (exit code: $EXIT_CODE)"
    echo ""
done

echo "=============================================="
echo "All frameworks completed!"
echo "Results saved to: $RESULTS_FILE"
echo "=============================================="
echo ""
echo "=== SUMMARY ==="
cat $RESULTS_FILE
