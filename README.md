Build docker:

docker image build -t llm\_elistratov:v1 .

Run: 

docker container run -p 8010:8010 llm\_elistratov:v1

Then send requests to the url.
