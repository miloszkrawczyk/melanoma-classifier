# Deploying to Azure Container Apps

What actually runs at
<https://skin-lesion-classifier.blackmoss-34728f66.germanywestcentral.azurecontainerapps.io>,
and how it got there. Written down because the commands live nowhere else, and
a deployment nobody can reproduce is a deployment that dies with the laptop
that made it.

Hosted on an Azure for Students subscription: $100 of credit, twelve months,
no card. Scaled to zero, so a demo nobody is using costs nothing.

## Once, to set up

```bash
az login
az group create --name melanoma --location germanywestcentral
```

A resource group is a box for everything else. It matters because deleting the
box deletes the contents, and there is no way to leave something running by
accident:

```bash
az group delete --name melanoma --yes    # removes all of it
```

Then the registry and the environment. `az containerapp up` creates both on
first run, but registering the providers by hand first avoids a long silent
retry loop:

```bash
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait
az provider register --namespace Microsoft.ContainerRegistry --wait
```

A fresh subscription has most services switched off. Miss one of these and the
CLI spins for twenty minutes generating log workspaces instead of saying what
is wrong.

## Building the image

Azure for Students cannot build images server-side. ACR Tasks is disabled on
free subscriptions, presumably because a free remote build service is a free
crypto miner, and it fails with `TasksOperationsNotAllowed`. So the image is
built locally and pushed:

```bash
az acr login --name ca0aa6078722acr

docker buildx build --platform linux/amd64 \
  -t ca0aa6078722acr.azurecr.io/skin-lesion-classifier:v2 \
  --push .
```

`--platform linux/amd64` matters on an Apple Silicon Mac. Containers are
portable across operating systems but not across processor architectures, and
an arm64 image will not start on Azure's x86 hosts. Emulated, the build takes
about fifteen minutes rather than two.

**Use a new tag every time.** Pushing new content under `:latest` leaves the
image reference unchanged, Azure sees nothing to do, and the old container
keeps running while the logs show code that is no longer in the repository.
That cost an hour to work out once already.

## Creating the app

```bash
PASSWORD=$(az acr credential show --name ca0aa6078722acr --query "passwords[0].value" -o tsv)

az containerapp create \
  --name skin-lesion-classifier \
  --resource-group melanoma \
  --environment skin-lesion-classifier-env \
  --image ca0aa6078722acr.azurecr.io/skin-lesion-classifier:v2 \
  --registry-server ca0aa6078722acr.azurecr.io \
  --registry-username ca0aa6078722acr \
  --registry-password "$PASSWORD" \
  --target-port 7860 \
  --ingress external \
  --cpu 1 --memory 2Gi \
  --min-replicas 0 --max-replicas 1
```

`--min-replicas 0` is what keeps this free. With no traffic the container is
shut down entirely and nothing is billed; a request wakes it, which takes
roughly a minute to start and load the weights. One replica kept warm instead
would cost something like $30-40 a month, which is most of the credit for a
page nobody is looking at.

## Updating it

```bash
docker buildx build --platform linux/amd64 \
  -t ca0aa6078722acr.azurecr.io/skin-lesion-classifier:v3 --push .

az containerapp update --name skin-lesion-classifier --resource-group melanoma \
  --image ca0aa6078722acr.azurecr.io/skin-lesion-classifier:v3
```

New tag, always. Then check a new revision actually took over:

```bash
az containerapp revision list --name skin-lesion-classifier --resource-group melanoma \
  --query "[].{revision:name, active:properties.active, image:properties.template.containers[0].image, state:properties.runningState}" -o table
```

## When it will not start

```bash
az containerapp logs show --name skin-lesion-classifier --resource-group melanoma --tail 40
```

This is the only thing that gives a real answer. Both failures during the first
deployment were found here in seconds after a lot of guessing: a permission
error from torchvision trying to download ImageNet weights at startup, and then
the same traceback again from the revision that was never replaced.
