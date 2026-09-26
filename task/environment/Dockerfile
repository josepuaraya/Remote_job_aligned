# Runtime image for the agent. Do not copy solution/, tests/test_outputs.py,
# hidden truth, or expected outputs into this image.
#
# This is the ONLY image the Gemini Data Collection (Nexus) sandbox runs. It
# ignores [verifier].environment_mode and executes the agent and the verifier in
# one container, so the verifier's own dependencies have to be installed here
# too. Without them tests/test.sh dies on "No module named pytest" and the task
# is rejected before a single check runs.
FROM --platform=linux/amd64 python:3.12-slim

ENV WORKSPACE_DIR=/workspace \
    DATA_DIR=/workspace/data \
    OUTPUT_DIR=/workspace/output \
    SOLUTION_DIR=/solution \
    TESTS_DIR=/tests \
    LOG_DIR=/logs/verifier

# Harbor sandbox providers inspect WORKDIR before Docker expands ENV values.
# Keep this literal so remote exec calls use the real container directory.
WORKDIR /workspace

# solve.py needs numpy (network inversion / linear algebra); the verifier
# needs pytest + the CTRF reporter. All three are vendored as Linux/amd64
# cp312 wheels under wheels/ and installed offline (--no-index
# --find-links), never fetched at build or run time. Keep this in step with
# tests/wheels/ (same pytest/ctrf versions) -- the sandbox is air-gapped, so
# anything solve.py or test.sh imports must be baked in at build time.
# Installing under /opt rather than /tmp is deliberate -- the runtime
# overlays /tmp with an empty tmpfs and would erase anything left there
# during the build.
COPY wheels/ /opt/wheels/

RUN python -m pip install --no-cache-dir --no-index \
        --find-links=/opt/wheels \
        -r /opt/wheels/requirements.txt \
    && rm -rf /opt/wheels

COPY data/ ${DATA_DIR}/

# The sandbox runs one container and executes both the agent and the verifier
# as this unprivileged user, so every directory either step writes to has to be
# owned by it. That includes the whole /logs tree the harness collects from:
# naming only ${LOG_DIR} leaves /logs itself root-owned, so the agent cannot
# create /logs/agent and its output redirect fails before the solution runs.
# The symptom is a trial that dies with no artifacts and no reward file.
RUN useradd --create-home --shell /bin/bash agent \
    && mkdir -p ${WORKSPACE_DIR} ${DATA_DIR} ${OUTPUT_DIR} ${SOLUTION_DIR} \
        ${TESTS_DIR} /logs /logs/agent /logs/artifacts ${LOG_DIR} \
    && chown -R agent:agent ${WORKSPACE_DIR} ${SOLUTION_DIR} ${TESTS_DIR} /logs

USER agent
