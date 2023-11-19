# FROM ubuntu:23.10
FROM python:3.9.17
COPY . /digits/

# RUN apt-get update
# RUN apt-get install -y python3 python3-pip
WORKDIR /digits/
RUN pip3 install -r requirements.txt
# CMD ["echo", "our first cmd line"]
ENV FLASK_APP=API/digit_prediction

EXPOSE 80

CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]
# CMD [" ","API/digit_prediction.py"]
# CMD ["pytest","&python","exp.py"]

# RUN pytest