export DAYLIGHT_SERVER_VERSION=$(cat __version__.py | cut -d '=' -f2 | xargs)
gcloud builds submit --region=europe-west1 --substitutions=_DAYLIGHT_SERVER_VERSION=${DAYLIGHT_SERVER_VERSION} --config imagebuild.yaml
# sed -r 's/DAYLIGHT_SERVER_VERSION/'"$DAYLIGHT_SERVER_VERSION"'/' imagebuild.yaml > container_versioned.yaml
gcloud run deploy daylight-server --image europe-north2-docker.pkg.dev/daylight-factor/daylight-server-docker-repo/daylight-server-img:${DAYLIGHT_SERVER_VERSION}