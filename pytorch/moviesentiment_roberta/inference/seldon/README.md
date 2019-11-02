Download model from Google Bucket:

```make download```

Wrap model into a docker image:

```make build```

Run under docker:

```make run_docker```

Note: you may need to wait for some time before sending requests as the model will be downloading
extra assets.

Test under docker:

```make test_docker```