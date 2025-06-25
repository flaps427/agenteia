import os
import pandas as pd
import logging
import sys
from dotenv import load_dotenv
import PySimpleGUI as sg
import threading  # Para evitar que a UI congele
import queue      # Para comunicação entre threads

# Importações do seu código original
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.prompts import PromptTemplate

# Suprimir avisos específicos da Langchain sobre código perigoso (se houver)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_experimental")

# Configurar logging para exibir mensagens no console (e pode ser direcionado para o GUI se desejado)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Variáveis globais para o agente e o dataframe para que sejam acessíveis por todas as funções GUI
agent_instance = None
df_instance = None
current_data_folder = None # Armazena o caminho da pasta de dados carregada

# Fila para comunicação inter-thread: (tipo_mensagem, conteúdo)
output_queue = queue.Queue()

# --- Funções do seu script original, adaptadas para logging e robustez ---
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
            logging.error(f"Erro: O caminho '{caminho_arquivos_extraidos}' não é um diretório válido.")
            return pd.DataFrame()
        for file in os.listdir(caminho_arquivos_extraidos):
            if file.lower().endswith('.csv'): # Usa .lower() para ser insensível a maiúsculas/minúsculas
                csv_files_encontrados.append(os.path.join(caminho_arquivos_extraidos, file))
    except FileNotFoundError:
        logging.error(f"Erro: A pasta '{caminho_arquivos_extraidos}' não foi encontrada.")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Erro ao listar arquivos na pasta '{caminho_arquivos_extraidos}': {e}")
        return pd.DataFrame()

    if len(csv_files_encontrados) < 2:
        logging.warning(f"Não foram encontrados CSVs suficientes (esperado 2) na pasta de destino. Encontrados: {len(csv_files_encontrados)}")
        return pd.DataFrame()

    arquivo_cabecalho_path = None
    arquivo_itens_path = None

    for csv_path in csv_files_encontrados:
        try:
            df_temp = pd.read_csv(csv_path, nrows=0, sep=',') # Lê apenas o cabeçalho
            colunas_do_arquivo = [col.upper() for col in df_temp.columns.tolist()] # Normaliza para maiúsculas
            
            chaves_cab_upper = [col.upper() for col in colunas_chave_cabecalho]
            chaves_itens_upper = [col.upper() for col in colunas_chave_itens]

            # Heurística para identificar arquivo de cabeçalho
            if all(col in colunas_do_arquivo for col in chaves_cab_upper) and \
               not any(col in colunas_do_arquivo for col in chaves_itens_upper):
                arquivo_cabecalho_path = csv_path
            # Heurística para identificar arquivo de itens
            elif all(col in colunas_do_arquivo for col in chaves_itens_upper) and \
                 not any(col in colunas_do_arquivo for col in chaves_cab_upper):
                arquivo_itens_path = csv_path

        except pd.errors.EmptyDataError:
            logging.warning(f"Aviso: O arquivo '{os.path.basename(csv_path)}' está vazio ou mal formatado.")
        except Exception as e:
            logging.warning(f"Aviso: Não foi possível inspecionar o arquivo '{os.path.basename(csv_path)}'. Erro: {e}")

    if not arquivo_cabecalho_path or not arquivo_itens_path:
        logging.error("Erro: Não foi possível identificar ambos os arquivos (cabeçalho e itens) pelos seus conteúdos de coluna. Verifique se os arquivos estão corretos e possuem as colunas esperadas.")
        return pd.DataFrame()

    # --- 2. Leitura dos arquivos CSV ---
    try:
        df_cabecalho = pd.read_csv(
            arquivo_cabecalho_path,
            sep=',',
            decimal='.',
            parse_dates=colunas_data_cabecalho
        )
        # Normaliza nomes de colunas para maiúsculas para o merge
        df_cabecalho.columns = [col.upper() for col in df_cabecalho.columns]
        logging.info(f"Lido arquivo de cabeçalho: {os.path.basename(arquivo_cabecalho_path)}")
    except Exception as e:
        logging.error(f"Erro ao ler o arquivo de cabeçalho '{arquivo_cabecalho_path}': {e}")
        return pd.DataFrame()

    try:
        df_itens = pd.read_csv(
            arquivo_itens_path,
            sep=',',
            decimal='.',
            parse_dates=colunas_data_itens
        )
        # Normaliza nomes de colunas para maiúsculas para o merge
        df_itens.columns = [col.upper() for col in df_itens.columns]
        logging.info(f"Lido arquivo de itens: {os.path.basename(arquivo_itens_path)}")
    except Exception as e:
        logging.error(f"Erro ao ler o arquivo de itens '{arquivo_itens_path}': {e}")
        return pd.DataFrame()
    
    # --- 3. Preparação para a união ---
    chaves_uniao_upper = [col.upper() for col in chaves_uniao]
    colunas_cabecalho_para_adicionar_upper = [col.upper() for col in colunas_cabecalho_para_adicionar]

    # Verifica se as chaves de união existem em ambos os DataFrames
    if not all(col in df_cabecalho.columns for col in chaves_uniao_upper):
        logging.error(f"Erro: As chaves de união {chaves_uniao_upper} não foram encontradas no arquivo de cabeçalho.")
        return pd.DataFrame()
    if not all(col in df_itens.columns for col in chaves_uniao_upper):
        logging.error(f"Erro: As chaves de união {chaves_uniao_upper} não foram encontradas no arquivo de itens.")
        return pd.DataFrame()

    colunas_do_cabecalho_existentes_e_unicas = [
        col for col in colunas_cabecalho_para_adicionar_upper if col in df_cabecalho.columns
    ]
    df_cabecalho_para_unir = df_cabecalho[chaves_uniao_upper + colunas_do_cabecalho_existentes_e_unicas]

    # --- 4. Realiza a união dos DataFrames ---
    try:
        df_unificado = pd.merge(
            df_itens,
            df_cabecalho_para_unir,
            on=chaves_uniao_upper,
            how='left'
        )
        logging.info("DataFrames de itens e cabeçalho unidos com sucesso.")
    except Exception as e:
        logging.error(f"Erro ao unir os DataFrames: {e}")
        return pd.DataFrame()

    # --- 5. Assegurando os formatos das colunas numéricas e datetime ---
    for col in colunas_numericas:
        if col.upper() in df_unificado.columns:
            df_unificado[col.upper()] = pd.to_numeric(df_unificado[col.upper()], errors='coerce')
    if coluna_datetime.upper() in df_unificado.columns:
        df_unificado[coluna_datetime.upper()] = pd.to_datetime(df_unificado[coluna_datetime.upper()], errors='coerce')
    
    # --- 6. Assegurando que todas as demais colunas estejam no formato string ---
    colunas_ja_tratadas = [c.upper() for c in colunas_numericas + [coluna_datetime]] 
    for col in df_unificado.columns:
        if col not in colunas_ja_tratadas and not pd.api.types.is_string_dtype(df_unificado[col]):
            df_unificado[col] = df_unificado[col].astype(str)

    logging.info(f"DataFrame unificado processado. Shape: {df_unificado.shape}")
    return df_unificado

def initialize_agent(data_folder_path: str):
    """
    Função que carrega os dados e inicializa o agente.
    Executada em uma thread separada para não bloquear a UI.
    """
    global df_instance, agent_instance, current_data_folder

    # Carrega a chave da API do arquivo .env
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(script_dir, '.env')
    load_dotenv(dotenv_path=dotenv_path)
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_api_key:
        output_queue.put(('error', "Chave GEMINI_API_KEY não encontrada no arquivo .env. Certifique-se de que ela está definida."))
        return False

    output_queue.put(('status', "Processando dados das notas fiscais..."))
    df = processar_dados_notas_fiscais_extraidos(data_folder_path)
    
    if df.empty:
        output_queue.put(('error', "Não foi possível carregar os dados. Verifique a pasta selecionada e os arquivos CSV."))
        return False

    df_instance = df
    current_data_folder = data_folder_path # Armazena a pasta carregada com sucesso

    # Define o prompt base para o agente
    template = """
    Você é um analista contábil.
    Sempre responda com base em dfs_csv, só responda sobre NF e item com base em dfs_csv.
    Não tente inventar uma resposta.

    Pergunta: {question}
    """
    prompt = PromptTemplate.from_template(template)
    
    # Configura e cria o agente
    base_url = 'https://generativelanguage.googleapis.com'
    model="gemini-1.5-flash" # Gemini 1.5 Flash é mais rápido e pode ser mais econômico
    try:
        llm = ChatGoogleGenerativeAI(base_url=base_url, model=model, google_api_key=gemini_api_key)
        agent_instance = create_pandas_dataframe_agent(
            llm, 
            df_instance, 
            verbose=True, 
            allow_dangerous_code=True,
            agent_executor_kwargs={'handle_parsing_errors':True},
            prompt = prompt
        )
        output_queue.put(('status', f"Agente 'Analisador de NF' pronto para a pasta: {os.path.basename(data_folder_path)}"))
        output_queue.put(('agent_ready', True)) # Sinaliza que o agente está pronto para habilitar a UI
        return True
    except Exception as e:
        output_queue.put(('error', f"Erro ao inicializar o agente: {e}"))
        return False

def invoke_agent(question: str):
    """
    Função que invoca o agente com a pergunta do usuário.
    Executada em uma thread separada para não bloquear a UI.
    """
    global agent_instance

    if agent_instance is None:
        output_queue.put(('error', "O agente não foi inicializado. Por favor, carregue os dados primeiro."))
        return

    output_queue.put(('status', "Processando sua pergunta..."))
    try:
        response = agent_instance.invoke({"input": question})
        output_queue.put(('agent_response', response.get('output', 'Nenhuma resposta encontrada.')))
        output_queue.put(('status', "Pronto para outra pergunta."))
    except Exception as e:
        output_queue.put(('error', f"Erro ao processar a pergunta: {e}"))
        output_queue.put(('status', "Pronto para outra pergunta."))

# --- Layout da Interface Gráfica PySimpleGUI ---
def create_gui_layout():
    sg.theme('LightBlue3') # Um tema moderno e elegante

    # Frame para a seleção de arquivos e carregamento de dados
    file_selection_frame = [
        [sg.Text("1. Selecione a pasta com os arquivos CSV extraídos:", tooltip="Esta pasta deve conter os arquivos CSV de Cabeçalho e Itens descompactados (não o arquivo ZIP).")],
        [sg.Input(key='-FOLDER_PATH-', enable_events=True, default_text="Selecione a pasta...", size=(50, 1), background_color='#ffffff'), 
         sg.FolderBrowse('Navegar', target='-FOLDER_PATH-', initial_folder=os.getcwd(), button_color=('white', '#007ACC')),
         sg.Button('Carregar Dados', key='-LOAD_DATA-', tooltip="Carrega os dados e inicializa o Analisador de NF.", button_color=('white', '#28a745'))
        ],
        [sg.Text("Status:", size=(8,1)), sg.Text("Aguardando seleção de pasta...", size=(60,1), key='-STATUS_TEXT-', text_color='blue')]
    ]

    # Frame para a conversa com o agente
    chat_frame = [
        [sg.Text("2. Faça uma pergunta ao Analisador de NF:", tooltip="Digite sua pergunta sobre as notas fiscais carregadas.")],
        # Área de histórico da conversa, simulando um chat
        [sg.Multiline(size=(80, 20), key='-CONVERSATION_HISTORY-', autoscroll=True, auto_refresh=True, 
                      enable_events=False, disabled=True, background_color='#e0e0e0', text_color='black',
                      font=('Helvetica', 10), pad=(5,5))], # padding para melhor visual
        # Campo de entrada para a pergunta e botão de envio
        [sg.Input(size=(70, 1), key='-QUERY_INPUT-', enable_events=True, font=('Helvetica', 12), background_color='#ffffff'), 
         sg.Button('Enviar', key='-SEND_QUERY-', bind_return_key=True, button_color=('white', '#007ACC'))]
    ]

    # Layout principal da janela
    layout = [
        [sg.Column([
            [sg.Text("Analisador de NF", font=('Helvetica', 28, 'bold'), justification='center', expand_x=True, text_color='#007ACC')],
            [sg.HSeparator(color='#007ACC')], # Separador horizontal
            [sg.Frame("Configuração de Dados", file_selection_frame, relief=sg.RELIEF_GROOVE, border_width=2, background_color='#f0f0f0')],
            [sg.Frame("Conversa com o Agente", chat_frame, relief=sg.RELIEF_GROOVE, border_width=2, background_color='#f8f8f8')],
            [sg.Text("Desenvolvido por Adapta", size=(80,1), justification='right', font=('Helvetica', 8, 'italic'), text_color='grey')]
        ], element_justification='center', vertical_alignment='top', expand_x=True, expand_y=True)]
    ]
    return layout

def main_gui():
    layout = create_gui_layout()
    # Cria a janela principal
    window = sg.Window("Analisador de NF - Chatbot", layout, resizable=True, finalize=True, element_justification='center') 

    # Desabilita o campo de pergunta e o botão de enviar inicialmente
    window['-QUERY_INPUT-'].update(disabled=True)
    window['-SEND_QUERY-'].update(disabled=True)

    while True:
        event, values = window.read(timeout=100) # Adiciona um timeout para checar a fila periodicamente

        if event == sg.WIN_CLOSED:
            break

        if event == '-LOAD_DATA-':
            folder_path = values['-FOLDER_PATH-']
            if not folder_path or not os.path.isdir(folder_path):
                window['-STATUS_TEXT-'].update("Por favor, selecione uma pasta válida.", text_color='red')
                continue
            
            # Desabilita os botões enquanto o processamento ocorre
            window['-LOAD_DATA-'].update(disabled=True)
            window['-FOLDER_PATH-'].update(disabled=True)
            window['-SEND_QUERY-'].update(disabled=True)
            window['-QUERY_INPUT-'].update(disabled=True)
            window['-STATUS_TEXT-'].update("Carregando e inicializando agente...", text_color='orange')

            # Inicia a inicialização do agente em uma thread separada
            threading.Thread(target=initialize_agent, args=(folder_path,), daemon=True).start()

        if event == '-SEND_QUERY-':
            # CORREÇÃO AQUI: Removido o .get()
            query = values['-QUERY_INPUT-'].strip() 
            if query:
                # Adiciona a pergunta do usuário ao histórico de conversa
                current_history = window['-CONVERSATION_HISTORY-'].get()
                window['-CONVERSATION_HISTORY-'].update(current_history + f"Você: {query}\n", append=False) 
                
                # Desabilita o campo de entrada e o botão de enviar
                window['-QUERY_INPUT-'].update('', disabled=True)
                window['-SEND_QUERY-'].update(disabled=True)
                window['-STATUS_TEXT-'].update("Enviando pergunta...", text_color='orange')

                # Inicia a invocação do agente em uma thread separada
                threading.Thread(target=invoke_agent, args=(query,), daemon=True).start()
            else:
                window['-STATUS_TEXT-'].update("Por favor, digite uma pergunta.", text_color='red')

        # Verifica a fila por mensagens das threads de trabalho
        try:
            while True: # Processa todos os itens na fila
                msg_type, msg_content = output_queue.get_nowait()
                if msg_type == 'status':
                    window['-STATUS_TEXT-'].update(msg_content, text_color='blue')
                elif msg_type == 'error':
                    window['-STATUS_TEXT-'].update(msg_content, text_color='red')
                    sg.popup_error("Erro", msg_content) # Mostra também um popup para erros críticos
                    # Reabilita os botões de carregamento se a inicialização do agente falhou
                    window['-LOAD_DATA-'].update(disabled=False)
                    window['-FOLDER_PATH-'].update(disabled=False)
                    window['-QUERY_INPUT-'].update(disabled=True)
                    window['-SEND_QUERY-'].update(disabled=True)
                elif msg_type == 'agent_response':
                    current_history = window['-CONVERSATION_HISTORY-'].get()
                    window['-CONVERSATION_HISTORY-'].update(current_history + f"Agente: {msg_content}\n\n", append=False)
                    # Reabilita o campo de entrada e o botão de enviar após a resposta
                    window['-QUERY_INPUT-'].update(disabled=False)
                    window['-SEND_QUERY-'].update(disabled=False)
                    window['-QUERY_INPUT-'].set_focus() # Volta o foco para o campo de entrada
                elif msg_type == 'agent_ready': # Sinaliza que o agente foi inicializado com sucesso
                    window['-QUERY_INPUT-'].update(disabled=False)
                    window['-SEND_QUERY-'].update(disabled=False)
                    window['-LOAD_DATA-'].update(disabled=False)
                    window['-FOLDER_PATH-'].update(disabled=False)
                    window['-QUERY_INPUT-'].set_focus()

        except queue.Empty:
            pass # Nenhuma mensagem na fila, continua

    window.close()

if __name__ == "__main__":
    main_gui()