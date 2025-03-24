# Базовый образ с Qdrant
FROM qdrant/qdrant

# Открываем порты для API и gRPC
EXPOSE 6333
EXPOSE 6334
