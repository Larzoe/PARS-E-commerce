# Thesis Project Lars

Deze github repository hoort de volgende bestanden te bevatten:
- main.py
- LSTM-test.keras
- tokenizer-test.pickle
- environment.yml

Voor de server aangezet wordt eerst even kijken of er nog nieuwe commits zijn in de repository.
```
git pull
```

Vanuit de environment.yml file can de juiste omgeving worden gecreeerd met behulp van de volgende prompt:
```
conda env create -n ENVNAME --file environment.yml
```

Hierna kan de applicatie aangezet worden met de volgende prompt:
```
python main.py
```

Als alles goed is worden in de cmd de sessie_id en de prediction geprint.
