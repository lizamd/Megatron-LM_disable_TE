set -x

cd ../cookbook/benchmarks/sizing/
echo "N=K=2048"
cat mm_m_range2048.txt | grep Thr | awk -F': ' '{print $2}'
echo "N=K=4096"
cat mm_m_range4096.txt | grep Thr | awk -F': ' '{print $2}'
echo "N=K=8192"
cat mm_m_range8192.txt | grep Thr | awk -F': ' '{print $2}'
