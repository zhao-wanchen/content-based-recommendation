# Content based recommendation of text items through various techniques

## About
This project uses the following techniques:
- Bidirectional Encoder Representations of Transformers (BERT);
- Global Vectors (GloVe)
- Sentence Transformers
- Universal Sentence Encoders

To create text embedding for:
- items in train_df, which represents potential candidates for recommendations;
- itmes in test_df, which represents user historical data and the basis for recommendation

Computes similarity scores. The final results include:
- Algorithm name
- Recommended items, comprehensive by item title, and respective similarity scores
- Overall recommendation performance metrics, including diveristy, personalization and coverage


## Building Docker image
```
docker build -t content-based-recommendation -f Dockerfile_content_based . --build-arg DB_HOST=<database-host> --build-arg DB_PSSW=<database-password> --build-arg DB_PORT=<database-port> --build-arg DB_USER=<database-username> --build-arg DB_SCHEMA=<database-schema>
```

```
docker run -d --name content-based-recommendation --entrypoint "/bin/bash" -it -p <port>:<port> content-based-recommendation
```

```
docker exec -it content-based-recommendation /bin/bash
```

```
python3 main.py
```