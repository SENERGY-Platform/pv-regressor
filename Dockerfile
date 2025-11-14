FROM python:3.10
LABEL org.opencontainers.image.source https://github.com/SENERGY-Platform/pv-regressor
WORKDIR /usr/src/app
COPY . .
RUN apt-get update && apt-get install -y git && apt-get install -y cmake
RUN apt-get install -y gfortran
RUN git log -1 --pretty=format:"commit=%H%ndate=%cd%n" > git_commit
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN apt-get purge -y git && apt-get auto-remove -y && apt-get clean && rm -rf .git
CMD [ "python3", "-u", "main.py"]