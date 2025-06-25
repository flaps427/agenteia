import os
import pandas as pd
import logging
import sys
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.prompts import PromptTemplate


# Variáveis de entrada
nome_arquivo_zip = '202401_NFs.zip'
caminho_pasta_destino = r'C:\Users\Aliander\Documents\Estudos IA\dados_extraidos'
arquivo_saida_nome = 'NFs_Unificadas.csv'

def processar_dados_notas_fiscais_extraidos(caminho_arquivos_extraidos: str) -> pd.DataFrame:
    # --- Configurações das colunas para identificação e processamento ---
    chaves_uniao = ['CHAVE DE ACESSO', 'MODELO', 'SÉRIE', 'NÚMERO']

    colunas_cabecalho_para_adicionar = [
        'EVENTO MAIS RECENTE',
        'DATA/HORA EVENTO MAIS RECENTE',
        'VALOR NOTA FISCAL'
    ]

    colunas_chave_cabecalho = ['VALOR NOTA FISCAL', 'EVENTO MAIS RECENTE']
    colunas_chave_itens = ['NÚMERO PRODUTO', 'QUANTIDADE', 'VALOR UNITÁRIO', 'VALOR TOTAL']

    colunas_data_cabecalho = ['DATA EMISSÃO', 'DATA/HORA EVENTO MAIS RECENTE']
    colunas_data_itens = ['DATA EMISSÃO']

    colunas_numericas = ['QUANTIDADE', 'VALOR UNITÁRIO', 'VALOR TOTAL', 'VALOR NOTA FISCAL']
    coluna_datetime = 'DATA/HORA EVENTO MAIS RECENTE'

    # --- 1. Identificação dos arquivos CSV ---
    csv_files_encontrados = []
    try:
        if not os.path.isdir(caminho_arquivos_extraidos):
            print(f"[FUNÇÃO]: Erro: O caminho '{caminho_arquivos_extraidos}' não é um diretório válido.")
            return pd.DataFrame()
        for file in os.listdir(caminho_arquivos_extraidos):
            if file.endswith('.csv'):
                csv_files_encontrados.append(os.path.join(caminho_arquivos_extraidos, file))
    except FileNotFoundError:
        print(f"[FUNÇÃO]: Erro: A pasta '{caminho_arquivos_extraidos}' não foi encontrada.")
        return pd.DataFrame()
    except Exception as e:
        print(f"[FUNÇÃO]: Erro ao listar arquivos na pasta '{caminho_arquivos_extraidos}': {e}")
        return pd.DataFrame()

    if len(csv_files_encontrados) < 2:
        print(f"[FUNÇÃO]: Erro: Não foram encontrados CSVs suficientes (esperado 2) na pasta de destino.")
        return pd.DataFrame()

    arquivo_cabecalho_path = None
    arquivo_itens_path = None

    for csv_path in csv_files_encontrados:
        try:
            df_temp = pd.read_csv(csv_path, nrows=0, sep=',')
            colunas_do_arquivo = df_temp.columns.tolist()

            if all(col in colunas_do_arquivo for col in colunas_chave_cabecalho) and \
               not any(col in colunas_do_arquivo for col in colunas_chave_itens):
                arquivo_cabecalho_path = csv_path
                # print(f"[FUNÇÃO]: Arquivo de Cabeçalho identificado: '{os.path.basename(csv_path)}'")
            elif all(col in colunas_do_arquivo for col in colunas_chave_itens) and \
                 not any(col in colunas_do_arquivo for col in colunas_chave_cabecalho):
                arquivo_itens_path = csv_path
                # print(f"[FUNÇÃO]: Arquivo de Itens identificado: '{os.path.basename(csv_path)}'")

        except Exception as e:
            print(f"[FUNÇÃO]: Aviso: Não foi possível inspecionar o arquivo '{os.path.basename(csv_path)}'. Erro: {e}")

    if not arquivo_cabecalho_path or not arquivo_itens_path:
        print("[FUNÇÃO]: Erro: Não foi possível identificar ambos os arquivos (cabeçalho e itens) pelos seus conteúdos de coluna.")
        return pd.DataFrame()

    # --- 2. Leitura dos arquivos CSV ---
    try:
        df_cabecalho = pd.read_csv(
            arquivo_cabecalho_path,
            sep=',',
            decimal='.',
            parse_dates=colunas_data_cabecalho
        )
    except Exception as e:
        print(f"[FUNÇÃO]: Erro ao ler o arquivo de cabeçalho '{arquivo_cabecalho_path}': {e}")
        return pd.DataFrame()

    try:
        df_itens = pd.read_csv(
            arquivo_itens_path,
            sep=',',
            decimal='.',
            parse_dates=colunas_data_itens
        )
    except Exception as e:
        print(f"[FUNÇÃO]: Erro ao ler o arquivo de itens '{arquivo_itens_path}': {e}")
        return pd.DataFrame()

    # --- 3. Preparação para a união ---
    colunas_do_cabecalho_existentes_e_unicas = [
        col for col in colunas_cabecalho_para_adicionar if col in df_cabecalho.columns
    ]
    df_cabecalho_para_unir = df_cabecalho[chaves_uniao + colunas_do_cabecalho_existentes_e_unicas]

    # --- 4. Realiza a união dos DataFrames ---
    df_unificado = pd.merge(
        df_itens,
        df_cabecalho_para_unir,
        on=chaves_uniao,
        how='left'
    )

    # --- 5. Assegurando os formatos das colunas numéricas e datetime ---
    for col in colunas_numericas:
        if col in df_unificado.columns:
            df_unificado[col] = pd.to_numeric(df_unificado[col], errors='coerce')
    if coluna_datetime in df_unificado.columns:
        df_unificado[coluna_datetime] = pd.to_datetime(df_unificado[coluna_datetime], errors='coerce')
    
    # --- 6. Assegurando que todas as demais colunas estejam no formato string ---
    colunas_ja_tratadas = colunas_numericas + [coluna_datetime]
    for col in df_unificado.columns:
        if col not in colunas_ja_tratadas and not pd.api.types.is_string_dtype(df_unificado[col]):
            df_unificado[col] = df_unificado[col].astype(str)

    return df_unificado


def main():
    # --- 1. Carrega as variáveis ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(script_dir, '.env')
    load_dotenv(dotenv_path=dotenv_path)
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    pergunta = ''
    
    if not  gemini_api_key:
        logging.critical("Chave GEMINI_API_KEY não encontrada no arquivo .env. Certifique-se de que ela está definida.")
        sys.exit(1)

    caminho_completo_zip = os.path.join(os.getcwd(), nome_arquivo_zip)
    caminho_arquivo_saida = os.path.join(caminho_pasta_destino, arquivo_saida_nome)
    df = processar_dados_notas_fiscais_extraidos(caminho_pasta_destino)
    
    # --- 2. Define o prompt base
    template = """
    Você é um analista contabil .
    Sempre responda com base em dfs_csv , só reponda sobre NF e item com base em dfs_csv .
    Não tente inventar uma resposta

    Pergunta: {question}
    """
    prompt = PromptTemplate.from_template(template)
    
    # --- 2. Configura e cria o agente ---
    base_url = 'https://generativelanguage.googleapis.com'
    model="gemini-2.5-flash"
    llm = ChatGoogleGenerativeAI(base_url=base_url, model=model,google_api_key=gemini_api_key)
    agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True,agent_executor_kwargs={'handle_parsing_errors':True},prompt = prompt )
    
    
    
    pergunta = "Quais os iten mais caro?"
    resposta = agent.invoke({"input": pergunta})

    print(f"Pergunta: {pergunta}")
    print(f"Resposta: {resposta.get('output')}")
    
    
if __name__ == "__main__":
    main()
