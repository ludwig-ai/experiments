cat /dev/null >autoscale.logs

while true
do
    kubectl logs -f ray-operator-569b7bb757-jd4n8 >>autoscale.logs
done

