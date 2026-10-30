# Humanoide 12 DOF

Visualizador 3D e base de controle dinâmico gerados a partir de `completo.step`.

## Rodar

```
pip install numpy aiohttp pyserial
python -m controle.servidor                  # http://127.0.0.1:8790, conecte a ESP32 pela pagina
python -m controle.servidor --serial COM5    # ja conecta (USB ou COM do Bluetooth); SIMULADO = emulador
```

### Executável

`dist/Humanoide.exe` (arquivo único, não precisa de Python): sobe o servidor e abre o visualizador numa janela
própria do Edge (ou Chrome); fechar a janela encerra o programa. Log em `%LOCALAPPDATA%\Humanoide\humanoide.log`.
Para gerar de novo após mudar o código: `python ferramentas/gerar_exe.py`.

Sem o servidor, o visualizador funciona em modo local (sliders movem só o modelo):
`python -m http.server 8791` e abra `http://127.0.0.1:8791/visualizador/`.

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `ferramentas/converter_step.py` | STEP → `modelo/` (malhas por elo, massas, inércias, juntas, URDF). Rode de novo se o CAD mudar. |
| `modelo/robo.json` | Árvore cinemática: elos (massa, COM, inércia) e juntas (origem, eixo, limites, torque). |
| `modelo/robo.urdf` + `meshes/` | Mesmo modelo para PyBullet / MuJoCo / ROS. |
| `config/poses.json` | Poses nomeadas (graus), usadas pelo visualizador e pelos controladores. |
| `controle/modelo.py` | Cinemática direta, centro de massa, jacobianos. |
| `controle/dinamica.py` | RNEA (dinâmica inversa), CRBA (matriz de massa), dinâmica direta, gravidade. |
| `controle/simulador.py` | Simulação com base fixa + modelo do servo MG90S (PD interno, saturação torque × velocidade). |
| `controle/controladores.py` | `Manual`, `Agachamento`, `Acenar`, `PDCompensacaoGravidade`. Novos controladores entram aqui. |
| `controle/hardware.py` | Driver da ESP32 (USB ou Bluetooth SPP): comandos com espera de ACK, telemetria, `JALL` a 50 Hz. |
| `controle/esp32_simulada.py` | Emulador do firmware (porta `SIMULADO`) para testar sem a placa. |
| `controle/servidor.py` | Laço de controle a 50 Hz + WebSocket para o visualizador + envio à ESP32. |
| `firmware/esp32_humanoide/` | PlatformIO, ESP32 + PCA9685 + MPU-6050, protocolo da bancada v1.1 + modo de configuração (v2.0). |

## Configurar os servos (ESP32)

Grave o firmware (`cd firmware/esp32_humanoide && pio run --target upload`), abra a página, escolha a porta e
**conectar**. Em **Configurar servos**:

1. **CFG**: libera a configuração; os limites ficam soltos e a velocidade cai para 60°/s.
2. Escolha o **canal**, clique **identificar** (balança ±8°) e veja qual junta mexe; escolha a **junta** e **associar**
   (a peça fica azul no modelo 3D).
3. Mova o slider **servo** até o fim de curso de um lado, espere parar, **min = atual**; idem **max = atual**;
   leve à posição da junta em 0° (pose do CAD: braços abertos, pernas retas) e **zero = atual**.
4. **Sentido**: se o servo gira ao contrário do modelo, use -1. **Vel.** limita graus/s (0 = livre). **ativo** desliga o pulso do canal.
5. **testar limites**, depois **salvar na ESP32** (grava na flash; **descartar** volta ao que está gravado) e **RUN**.

Em RUN a ESP32 recusa `POS` fora de [min, max] (`ERR:LIMIT:min:max`) e satura `J`/`JALL` nos limites
(canal aparece em `clamped` no STATUS). Ao conectar, os limites gravados viram os limites das juntas no simulador e
nos sliders. Os comandos estão no cabeçalho de `firmware/esp32_humanoide/src/main.cpp`.

## Convenções

- Frame: X frente, Y esquerda do robô, Z para cima; metros, kg, radianos (graus só na UI/poses).
- Juntas: `quadril_roll`, `quadril_pitch`, `joelho`, `tornozelo` (cada perna) e `ombro`, `cotovelo` (cada braço), sufixo `_L`/`_R`.
- Pitch gira em +Y para os dois lados: agachar = quadril `-a`, joelho `+2a`, tornozelo `-a`.
- `sentido_servo` no `robo.json` = sinal entre o eixo canônico e o eixo de saída do servo no CAD (palpite inicial para o **Sentido** na ESP32).

## Premissas a conferir

- Divisão das peças entre elos: regras em `elo_da_peca()` do conversor (ex.: os suportes do ombro ficaram no torso).
- Massas: servos 13,4 g, bateria 18650 46 g, ESP32 10 g, PCA9685 9 g, tela 5 g, impressos 0,8 g/cm³ (`--densidade`). Total ≈ 511 g; pese o robô e ajuste.
- Servo: torque 0,21 N·m, 10,5 rad/s, inércia refletida 1e-3 kg·m² (estimativa). Limites ±90° em todas as juntas.
- A dinâmica é de base fixa (torso preso). Para andar, com base flutuante e contato com o chão, use o `robo.urdf` no MuJoCo/PyBullet ou estenda o `dinamica.py`.
