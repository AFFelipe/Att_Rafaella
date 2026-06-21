from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, List
from datetime import datetime
from pathlib import Path
import joblib
import pandas as pd

MODELO_PATH = Path(__file__).with_name("modelo_final.joblib")
FEATURES_PATH = Path(__file__).with_name("colunas_features.joblib")

modelo = None
colunas_features = None
startup_error = None


def carregar_modelo():
    global modelo, colunas_features, startup_error

    if modelo is not None and colunas_features is not None:
        return

    if not MODELO_PATH.exists() or not FEATURES_PATH.exists():
        startup_error = (
            f"Arquivos do modelo não encontrados. "
            f"Coloque '{MODELO_PATH.name}' e '{FEATURES_PATH.name}' "
            f"na mesma pasta de '{Path(__file__).name}'."
        )
        return

    try:
        modelo = joblib.load(MODELO_PATH)
        colunas_features = joblib.load(FEATURES_PATH)
        startup_error = None
    except Exception as erro:
        startup_error = f"Erro ao carregar o modelo: {erro}"


carregar_modelo()

app = FastAPI(
    title="API de Predição de TDI - Anos Finais (Ensino Fundamental, PE)",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup():
    carregar_modelo()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

historico_predicoes: List[dict] = []


class PredicaoInput(BaseModel):
    tdi_anos_iniciais: float = Field(
        ..., ge=0, le=100,
        description="TDI do Ensino Fundamental - Anos Iniciais (%)",
    )
    localizacao: Literal["Urbana", "Rural"]
    dependencia: Literal["Estadual", "Municipal"]


class PredicaoOutput(BaseModel):
    tdi_anos_finais_previsto: float
    entrada: PredicaoInput
    timestamp: str


@app.post("/predict", response_model=PredicaoOutput)
def predict(dados: PredicaoInput):
    try:
        carregar_modelo()
        if startup_error is not None or modelo is None or colunas_features is None:
            raise HTTPException(
                status_code=503,
                detail=startup_error or "Modelo ainda não está disponível.",
            )

        linha = {
            "TDI_AI": dados.tdi_anos_iniciais,
            "LOCALIZACAO_Rural": 1 if dados.localizacao == "Rural" else 0,
            "DEP_Municipal": dados.dependencia == "Municipal",
        }

        X = pd.DataFrame(
            [[linha[coluna] for coluna in colunas_features]],
            columns=colunas_features,
        )

        predicao = round(float(modelo.predict(X)[0]), 2)

        resultado = PredicaoOutput(
            tdi_anos_finais_previsto=predicao,
            entrada=dados,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

        historico_predicoes.append(resultado.model_dump())

        return resultado

    except HTTPException:
        raise
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar predição: {erro}")


@app.get("/predictions")
def listar_historico():
    return {"total": len(historico_predicoes), "historico": historico_predicoes}
