# Multi-Channel Signal Analysis and Baroreflex Sensitivity

Este documento descreve os novos recursos adicionados ao BioSPPy para análise simultânea de múltiplos canais de sinais biosignais e análise de sensibilidade barorreflexo.

## Índice

- [Introdução](#introdução)
- [Instalação](#instalação)
- [Módulo MultiChannelSignal](#módulo-multichannelsignal)
- [Análise de Baroreflexo](#análise-de-baroreflexo)
- [Exemplos de Uso](#exemplos-de-uso)
- [API Reference](#api-reference)

## Introdução

Os novos módulos `multichannel` e `baroreflex` permitem:

1. **Importar e gerenciar múltiplos canais** de sinais simultaneamente (ECG, pressão arterial, PPG, respiração, etc.)
2. **Sincronizar canais temporalmente** usando correlação cruzada
3. **Analisar cada canal isoladamente** ou em conjunto
4. **Calcular sensibilidade barorreflexo** (BRS) combinando sinais de ECG e pressão arterial

### Funcionalidades Principais

- ✅ Gerenciamento de múltiplos canais com diferentes tipos de sinais
- ✅ Sincronização automática entre canais via correlação cruzada
- ✅ Processamento individual ou em lote de todos os canais
- ✅ Análise de sensibilidade barorreflexo (3 métodos: sequência, espectral, alfa)
- ✅ Alinhamento temporal e resampling de sinais
- ✅ Extração de taxa cardíaca de múltiplas fontes

## Instalação

Os novos módulos estão integrados ao BioSPPy. Após clonar o repositório:

```bash
pip install -e .
```

Ou, se já tiver o BioSPPy instalado, certifique-se de ter a versão mais recente:

```bash
git pull origin main
pip install -e . --upgrade
```

## Módulo MultiChannelSignal

### Conceito

A classe `MultiChannelSignal` gerencia múltiplos canais de sinais biosignais, permitindo:
- Adicionar diferentes tipos de sinais (ECG, ABP, PPG, respiração, EDA)
- Sincronizar canais automaticamente
- Processar cada canal com algoritmos específicos
- Extrair e combinar informações de múltiplos canais

### Uso Básico

```python
from biosppy.signals import multichannel
from biosppy import storage

# Carregar sinais
ecg_signal, _ = storage.load_txt('ecg.txt')
abp_signal, _ = storage.load_txt('abp.txt')

# Criar objeto multi-canal
mc = multichannel.MultiChannelSignal(sampling_rate=1000.0)

# Adicionar canais
mc.add_channel('ECG', ecg_signal, channel_type='ecg')
mc.add_channel('ABP', abp_signal, channel_type='abp')

# Sincronizar canais
offsets = mc.synchronize(reference_channel='ECG')

# Processar todos os canais
results = mc.process_all()

# Acessar resultados individuais
ecg_results = mc.get_processed('ECG')
print(f"R-peaks detectados: {len(ecg_results['rpeaks'])}")
print(f"Frequência cardíaca média: {np.mean(ecg_results['heart_rate']):.2f} bpm")
```

### Função de Conveniência

```python
# Criar e processar em uma única chamada
mc = multichannel.multichannel(
    signals={'ECG': ecg_signal, 'ABP': abp_signal},
    sampling_rate=1000.0,
    channel_types={'ECG': 'ecg', 'ABP': 'abp'},
    process=True,
    synchronize=True
)
```

### Tipos de Sinais Suportados

| Tipo | Descrição | Processamento |
|------|-----------|---------------|
| `'ecg'` | Eletrocardiograma | Detecção de picos R, templates, taxa cardíaca |
| `'abp'` | Pressão arterial | Detecção de onsets de pulso, taxa cardíaca |
| `'ppg'` | Fotopletismografia | Detecção de picos, templates, taxa cardíaca |
| `'resp'` | Respiração | Detecção de cruzamentos zero, taxa respiratória |
| `'eda'` | Atividade eletrodérmica | Filtragem e análise de características |

## Análise de Baroreflexo

### Conceito

O baroreflexo é um mecanismo de controle cardiovascular que regula a pressão arterial através de mudanças na frequência cardíaca. A sensibilidade barorreflexo (BRS) quantifica essa relação.

**BRS = Δ RR-interval / Δ Pressão Sistólica**

Valores típicos:
- **Adultos saudáveis**: 10-30 ms/mmHg
- **Reduzida**: < 10 ms/mmHg (indica menor função barorreflexo)
- **Aumentada**: > 30 ms/mmHg

### Métodos de Cálculo

#### 1. Método de Sequência (Bertinieri et al., 1988)

Identifica sequências espontâneas de aumentos ou diminuições simultâneas na pressão arterial e intervalos RR.

```python
from biosppy.signals import baroreflex

brs = baroreflex.baroreflex_sensitivity(
    rri=rri_array,  # Intervalos RR em ms
    sbp=sbp_array,  # Pressão sistólica em mmHg
    method='sequence',
    min_sequences=3
)

print(f"BRS (método sequência): {brs['brs_sequence']:.2f} ms/mmHg")
print(f"Sequências detectadas: {brs['n_sequences_up'] + brs['n_sequences_down']}")
```

**Parâmetros ajustáveis:**
- `sequence_length`: Comprimento mínimo da sequência (default: 3)
- `sequence_threshold`: Limiar para mudanças (default: 1.0)
- `sequence_delay`: Atraso máximo entre mudanças (default: 1)

#### 2. Método Espectral (Função de Transferência)

Calcula a função de transferência entre pressão arterial e intervalos RR no domínio da frequência.

```python
brs = baroreflex.baroreflex_sensitivity(
    rri=rri_array,
    sbp=sbp_array,
    method='spectral'
)

print(f"BRS LF: {brs['brs_spectral_lf']:.2f} ms/mmHg")
print(f"BRS HF: {brs['brs_spectral_hf']:.2f} ms/mmHg")
print(f"Coerência LF: {brs['coherence_lf']:.3f}")
print(f"Coerência HF: {brs['coherence_hf']:.3f}")
```

**Bandas de frequência:**
- **LF (Low Frequency)**: 0.04-0.15 Hz
- **HF (High Frequency)**: 0.15-0.4 Hz

#### 3. Método Alfa (Coeficiente α)

Calcula a raiz quadrada da razão entre potências espectrais de RRI e pressão arterial.

```python
brs = baroreflex.baroreflex_sensitivity(
    rri=rri_array,
    sbp=sbp_array,
    method='alpha'
)

print(f"Alpha LF: {brs['brs_alpha_lf']:.2f} ms/mmHg")
print(f"Alpha HF: {brs['brs_alpha_hf']:.2f} ms/mmHg")
```

#### Todos os Métodos

```python
# Calcular com todos os métodos
brs = baroreflex.baroreflex_sensitivity(
    rri=rri_array,
    sbp=sbp_array,
    method='all',
    show=True  # Mostra gráficos
)
```

### Análise Integrada com MultiChannelSignal

```python
from biosppy.signals import multichannel, baroreflex

# Criar e processar multi-canal
mc = multichannel.multichannel(
    signals={'ECG': ecg_signal, 'ABP': abp_signal},
    sampling_rate=1000.0,
    channel_types={'ECG': 'ecg', 'ABP': 'abp'},
    process=True
)

# Analisar baroreflexo diretamente
brs_results = baroreflex.analyze_multichannel_baroreflex(
    mc,
    ecg_channel='ECG',
    abp_channel='ABP',
    method='all'
)

print(f"BRS (sequência): {brs_results['brs_sequence']:.2f} ms/mmHg")
```

## Exemplos de Uso

### Exemplo 1: Análise Básica Multi-Canal

```python
from biosppy.signals import multichannel
from biosppy import storage
import numpy as np

# Carregar sinais
ecg, _ = storage.load_txt('./examples/ecg.txt')
resp, _ = storage.load_txt('./examples/resp.txt')

# Criar objeto multi-canal
mc = multichannel.MultiChannelSignal(sampling_rate=1000.0)
mc.add_channel('ECG', ecg, channel_type='ecg')
mc.add_channel('RESP', resp[:len(ecg)], channel_type='resp')

# Processar
mc.process_channel('ECG')
mc.process_channel('RESP')

# Extrair frequência cardíaca
ecg_results = mc.get_processed('ECG')
hr = ecg_results['heart_rate']
hr_ts = ecg_results['heart_rate_ts']

print(f"FC média: {np.mean(hr):.2f} bpm")
print(f"FC min/max: {np.min(hr):.2f} / {np.max(hr):.2f} bpm")
```

### Exemplo 2: Sincronização de Canais

```python
# Adicionar canais com offsets conhecidos
mc = multichannel.MultiChannelSignal(sampling_rate=1000.0)
mc.add_channel('ECG', ecg_signal, offset=0.0)
mc.add_channel('ABP', abp_signal, offset=0.1)  # 100 ms de offset

# Sincronizar automaticamente
offsets = mc.synchronize(method='cross_correlation')
print("Offsets detectados:", offsets)

# Obter vetores de tempo alinhados
time_vectors = mc.get_time_vector()
```

### Exemplo 3: Análise Completa de Baroreflexo

```python
from biosppy.signals import multichannel, baroreflex
import matplotlib.pyplot as plt

# Carregar e processar
mc = multichannel.multichannel(
    signals={'ECG': ecg_data, 'ABP': abp_data},
    sampling_rate=1000.0,
    channel_types={'ECG': 'ecg', 'ABP': 'abp'}
)

# Análise de baroreflexo
brs = baroreflex.analyze_multichannel_baroreflex(
    mc,
    method='all',
    show=False
)

# Visualizar resultados
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: RRI vs SBP
axes[0, 0].scatter(brs['sbp'], brs['rri'], alpha=0.5)
axes[0, 0].set_xlabel('Pressão Sistólica (mmHg)')
axes[0, 0].set_ylabel('Intervalo RR (ms)')
axes[0, 0].set_title('Relação RRI-SBP')

# Plot 2: BRS por método
methods = []
values = []
if not np.isnan(brs['brs_sequence']):
    methods.append('Sequência')
    values.append(brs['brs_sequence'])
if not np.isnan(brs['brs_spectral_lf']):
    methods.append('Espectral LF')
    values.append(brs['brs_spectral_lf'])

axes[0, 1].bar(methods, values)
axes[0, 1].set_ylabel('BRS (ms/mmHg)')
axes[0, 1].set_title('Sensibilidade Barorreflexo')

# Plot 3: Função de transferência
if 'transfer_function_freqs' in brs.keys():
    axes[1, 0].plot(brs['transfer_function_freqs'],
                     brs['transfer_function_gain'])
    axes[1, 0].set_xlabel('Frequência (Hz)')
    axes[1, 0].set_ylabel('Ganho (ms/mmHg)')
    axes[1, 0].set_title('Função de Transferência')
    axes[1, 0].set_xlim([0, 0.5])

# Plot 4: Coerência
if 'coherence' in brs.keys():
    axes[1, 1].plot(brs['transfer_function_freqs'], brs['coherence'])
    axes[1, 1].axhline(y=0.5, color='r', linestyle='--', label='Limiar')
    axes[1, 1].set_xlabel('Frequência (Hz)')
    axes[1, 1].set_ylabel('Coerência')
    axes[1, 1].set_title('Coerência')
    axes[1, 1].set_xlim([0, 0.5])
    axes[1, 1].legend()

plt.tight_layout()
plt.show()
```

### Exemplo 4: Análise em Lote

```python
# Processar múltiplos arquivos
import glob

results_summary = []

for ecg_file in glob.glob('data/ecg_*.txt'):
    subject_id = ecg_file.split('_')[-1].split('.')[0]
    abp_file = f'data/abp_{subject_id}.txt'

    # Carregar
    ecg, _ = storage.load_txt(ecg_file)
    abp, _ = storage.load_txt(abp_file)

    # Processar
    mc = multichannel.multichannel(
        signals={'ECG': ecg, 'ABP': abp},
        sampling_rate=1000.0,
        channel_types={'ECG': 'ecg', 'ABP': 'abp'}
    )

    # Analisar baroreflexo
    brs = baroreflex.analyze_multichannel_baroreflex(mc, method='sequence')

    results_summary.append({
        'subject': subject_id,
        'brs': brs['brs_sequence'],
        'n_sequences': brs['n_sequences_up'] + brs['n_sequences_down']
    })

# Resumo
import pandas as pd
df = pd.DataFrame(results_summary)
print(df.describe())
```

## API Reference

### MultiChannelSignal

#### Classe Principal

```python
class MultiChannelSignal(sampling_rate=1000.0)
```

**Métodos:**

- `add_channel(name, signal, channel_type=None, offset=0.0)` - Adiciona um canal
- `get_channel(name)` - Obtém dados do canal
- `list_channels()` - Lista todos os canais
- `synchronize(reference_channel=None, method='cross_correlation')` - Sincroniza canais
- `process_channel(name, **kwargs)` - Processa um canal específico
- `process_all(**kwargs)` - Processa todos os canais
- `get_processed(name)` - Obtém resultados processados
- `get_heart_rate_signals()` - Extrai sinais de frequência cardíaca
- `get_time_vector(name=None)` - Obtém vetor(es) de tempo
- `align_channels(target_length=None)` - Alinha canais ao mesmo comprimento
- `resample_channel(name, target_rate)` - Reamostrar um canal

#### Função de Conveniência

```python
multichannel(signals, sampling_rate=1000.0, channel_types=None,
             channel_names=None, process=True, synchronize=True, **kwargs)
```

**Parâmetros:**
- `signals` - Lista ou dicionário de sinais
- `sampling_rate` - Frequência de amostragem (Hz)
- `channel_types` - Lista ou dicionário de tipos de canal
- `channel_names` - Lista de nomes (se signals for lista)
- `process` - Processar canais automaticamente
- `synchronize` - Sincronizar canais automaticamente

### Baroreflex

#### Função Principal

```python
baroreflex_sensitivity(rri=None, sbp=None, rpeaks=None, systolic_peaks=None,
                       sampling_rate=1000.0, method='sequence',
                       min_sequences=3, show=False, **kwargs)
```

**Parâmetros:**
- `rri` - Intervalos RR (ms)
- `sbp` - Valores de pressão sistólica (mmHg)
- `rpeaks` - Localizações dos picos R (amostras)
- `systolic_peaks` - Localizações dos picos sistólicos (amostras)
- `sampling_rate` - Frequência de amostragem (Hz)
- `method` - Método: 'sequence', 'spectral', 'alpha', ou 'all'
- `min_sequences` - Mínimo de sequências para método de sequência
- `show` - Mostrar gráficos

**Retorna:**
Dicionário com:
- `brs_sequence` - BRS pelo método de sequência
- `brs_spectral_lf/hf` - BRS espectral nas bandas LF/HF
- `brs_alpha_lf/hf` - Coeficiente alfa nas bandas LF/HF
- `n_sequences_up/down` - Número de sequências
- `coherence_lf/hf` - Coerência nas bandas LF/HF
- `transfer_function_freqs/gain` - Função de transferência
- `coherence` - Vetor de coerência completo

#### Análise Integrada

```python
analyze_multichannel_baroreflex(mc_signal, ecg_channel='ECG',
                                abp_channel='ABP', method='all', **kwargs)
```

**Parâmetros:**
- `mc_signal` - Objeto MultiChannelSignal processado
- `ecg_channel` - Nome do canal ECG
- `abp_channel` - Nome do canal ABP
- `method` - Método de cálculo
- `**kwargs` - Argumentos adicionais para `baroreflex_sensitivity()`

## Notas de Uso

### Qualidade do Sinal

- Certifique-se de que os sinais estão adequadamente filtrados
- Use os módulos de avaliação de qualidade do BioSPPy quando disponíveis
- Sinais ruidosos podem afetar significativamente os resultados

### Sincronização

- Se os sinais foram gravados simultaneamente no mesmo sistema, a sincronização pode não ser necessária
- Para gravações não-simultâneas, use o método de correlação cruzada
- Offsets manuais podem ser especificados se conhecidos

### Análise de Baroreflexo

- **Duração recomendada**: 3-5 minutos de dados de boa qualidade
- **Condições**: Sujeito em repouso, posição supina
- **Interpretação clínica**: Sempre consulte profissionais de saúde
- **Valores típicos**: 5-30 ms/mmHg em adultos saudáveis
- Valores baixos (< 10 ms/mmHg) indicam função barorreflexo reduzida

### Limitações

- O método de sequência requer mudanças espontâneas significativas na PA
- O método espectral requer coerência suficiente (> 0.5) entre sinais
- Artefatos e ectopias podem afetar os resultados
- Use sempre detecção e correção de artefatos antes da análise

## Referências

1. Bertinieri, G., et al. (1988). "Evaluation of baroreceptor reflex by blood pressure monitoring in unanesthetized cats." *American Journal of Physiology*.

2. Laude, D., et al. (2004). "Comparison of various techniques used to estimate spontaneous baroreflex sensitivity." *American Journal of Physiology*.

3. Parati, G., et al. (2000). "Point:Counterpoint: Cardiovascular variability is/is not an index of autonomic control of circulation." *Journal of Applied Physiology*.

4. Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology (1996). "Heart rate variability: standards of measurement, physiological interpretation and clinical use." *Circulation*.

## Contribuindo

Para reportar bugs ou sugerir melhorias, abra uma issue no [repositório do BioSPPy](https://github.com/scientisst/BioSPPy).

## Licença

BSD 3-Clause License

Copyright (c) 2015-2025, Instituto de Telecomunicações

## Autores

Desenvolvido pela equipe do BioSPPy no Instituto de Telecomunicações, Portugal.

---

Para mais informações sobre o BioSPPy, visite: https://github.com/scientisst/BioSPPy
