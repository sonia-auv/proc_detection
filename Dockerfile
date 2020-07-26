ARG BASE_IMAGE_ARM="docker.pkg.github.com/sonia-auv/sonia_messages/sonia_messages:arm64-perception-l4t-latest"
ARG BASE_IMAGE_X86="docker.pkg.github.com/sonia-auv/sonia_messages/sonia_messages:arm64-perception-latest"
# LOCAL DEV ENV
FROM ${BASE_IMAGE_X86} as development-env

USER root

ARG BUILD_DATE
ARG VERSION

ENV NODE_NAME=proc_detection

LABEL net.etsmtl.sonia-auv.node.build-date=${BUILD_DATE}
LABEL net.etsmtl.sonia-auv.node.version=${VERSION}
LABEL net.etsmtl.sonia-auv.node.name=${NODE_NAME}


ENV SONIA_WS=${SONIA_HOME}/ros_sonia_ws

ENV NODE_NAME=${NODE_NAME}
ENV NODE_PATH=${SONIA_WS}/src/${NODE_NAME}
ENV LAUNCH_FILE=${NODE_NAME}.launch
ENV SCRIPT_DIR=${SONIA_WS}/scripts
ENV ENTRYPOINT_FILE=sonia_entrypoint.sh
ENV LAUNCH_ABSPATH=${NODE_PATH}/launch/${LAUNCH_FILE}
ENV ENTRYPOINT_ABSPATH=${NODE_PATH}/scripts/${ENTRYPOINT_FILE}

ENV SONIA_WS_SETUP=${SONIA_WS}/devel/setup.bash

WORKDIR ${SONIA_WS}

COPY . ${NODE_PATH}

RUN apt-get install python-pip

FROM development-env as development-env-no-gpu

RUN pip install ${NODE_PATH}/requirements/dev/requirements-no-gpu.txt

RUN bash -c "source ${ROS_WS_SETUP}; source ${BASE_LIB_WS_SETUP}; catkin_make"

RUN chown -R ${SONIA_USER}: ${SONIA_WS}
USER ${SONIA_USER}

RUN mkdir ${SCRIPT_DIR}
RUN cat $ENTRYPOINT_ABSPATH > ${SCRIPT_DIR}/entrypoint.sh
RUN echo "roslaunch --wait $LAUNCH_ABSPATH" > ${SCRIPT_DIR}/launch.sh

RUN chmod +x ${SCRIPT_DIR}/entrypoint.sh && chmod +x ${SCRIPT_DIR}/launch.sh

RUN echo "source $SONIA_WS_SETUP" >> ~/.bashrc

ENTRYPOINT ["./scripts/entrypoint.sh"]
CMD ["./scripts/launch.sh"]



FROM development-env as development-env-gpu

RUN pip install ${NODE_PATH}/requirements/dev/requirements-gpu.txt

RUN bash -c "source ${ROS_WS_SETUP}; source ${BASE_LIB_WS_SETUP}; catkin_make"

RUN chown -R ${SONIA_USER}: ${SONIA_WS}
USER ${SONIA_USER}

RUN mkdir ${SCRIPT_DIR}
RUN cat $ENTRYPOINT_ABSPATH > ${SCRIPT_DIR}/entrypoint.sh
RUN echo "roslaunch --wait $LAUNCH_ABSPATH" > ${SCRIPT_DIR}/launch.sh

RUN chmod +x ${SCRIPT_DIR}/entrypoint.sh && chmod +x ${SCRIPT_DIR}/launch.sh

RUN echo "source $SONIA_WS_SETUP" >> ~/.bashrc

ENTRYPOINT ["./scripts/entrypoint.sh"]
CMD ["./scripts/launch.sh"]

# JESTON XAVIER IMAGE
FROM ${BASE_IMAGE_ARM} as production-env

USER root

ARG BUILD_DATE
ARG VERSION

ENV NODE_NAME=proc_detection

LABEL net.etsmtl.sonia-auv.node.build-date=${BUILD_DATE}
LABEL net.etsmtl.sonia-auv.node.version=${VERSION}
LABEL net.etsmtl.sonia-auv.node.name=${NODE_NAME}


ENV SONIA_WS=${SONIA_HOME}/ros_sonia_ws

ENV NODE_NAME=${NODE_NAME}
ENV NODE_PATH=${SONIA_WS}/src/${NODE_NAME}
ENV LAUNCH_FILE=${NODE_NAME}.launch
ENV SCRIPT_DIR=${SONIA_WS}/scripts
ENV ENTRYPOINT_FILE=sonia_entrypoint.sh
ENV LAUNCH_ABSPATH=${NODE_PATH}/launch/${LAUNCH_FILE}
ENV ENTRYPOINT_ABSPATH=${NODE_PATH}/scripts/${ENTRYPOINT_FILE}

ENV SONIA_WS_SETUP=${SONIA_WS}/devel/setup.bash

WORKDIR ${SONIA_WS}

RUN pip install ${NODE_PATH}/requirements/prod/requirements.txt

COPY . ${NODE_PATH}
RUN bash -c "source ${ROS_WS_SETUP}; source ${BASE_LIB_WS_SETUP}; catkin_make"

RUN chown -R ${SONIA_USER}: ${SONIA_WS}
USER ${SONIA_USER}

RUN mkdir ${SCRIPT_DIR}
RUN cat $ENTRYPOINT_ABSPATH > ${SCRIPT_DIR}/entrypoint.sh
RUN echo "roslaunch --wait $LAUNCH_ABSPATH" > ${SCRIPT_DIR}/launch.sh

RUN chmod +x ${SCRIPT_DIR}/entrypoint.sh && chmod +x ${SCRIPT_DIR}/launch.sh

RUN echo "source $SONIA_WS_SETUP" >> ~/.bashrc

ENTRYPOINT ["./scripts/entrypoint.sh"]
CMD ["./scripts/launch.sh"]
