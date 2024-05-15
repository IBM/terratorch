## Submitting tasks to MCAD cluster

To submit a tuning task to an OpenShift cluster with MCAD, you can use a command, such as the following torchx command:

```torchx run --workspace "" --scheduler kubernetes_mcad --scheduler_args namespace="geospatial-alpha" utils.sh --image="us.icr.io/gfmaas/terratorch:dev1" --gpu 1 --memMB 24000 --mounts type=volume,src=geoft-service-datasets-pvc,dst="/data",type=volume,src=test-temp-pvc,dst="/working",type=volume,src=gfm-models-pvc,dst="/terratorch/gfm_models" -- terratorch fit -c /working/sen1floods11_swin_studio.yaml```

