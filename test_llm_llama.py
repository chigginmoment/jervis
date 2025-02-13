from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

template = """
<｜begin▁of▁sentence｜>
You are a helpful digital assistant. You provide answers as accurately as you can, but you never lie.
If unsure at all about something, you answer "I am unsure".
You are tasked with receiving context and extracting the information that is relevant to the user's query.
Think about what the user is asking and the best way to understand the context before constructing an answer.
Do not rely on numbers to formulate your answer, only on the document's statements. Numbers are relative.
Once you're done thinking, answer in a single sentence as concisely as possible. If the user query is unrelated
to your provided context, refuse to answer the question.
You are not asked to extrapolate any hidden information from the context. Simply state the most relevant point.

You will mainly be reading wiki pages. Here are some helpful hints:
# For World of Tanks wiki pages:
- The penetration format is in [standard round penetration]/[premium round penetration]/[explosive round penetration] format, in millimeters.
- When reading about stats in the Performance section, they are often presented backwards. E.g. 6,100,000\nCost means Cost: 6,100,000.

<context>
VIII 9650 The Bofors Tornvagn is a Swedish tier 8 premium heavy tank . One of the projects of the Bofors company in the late 1970s. The vehicle was supposed to incorporate several bold technical solutions. For example, they planned to mount a complex, separate loading system: Projectiles were placed under the turret on the right side, and charges were fed from the rear by a belt system. The vehicle was supposed to feature impressive protection for its time, but no real metal prototype was ever built.
 Cost 9,650
 Hit Points 320 HP
 Weight Limit 23.37/45 t
 Engine Power 500 hp
 Speed Limit 32/12 km/h
 Traverse 22 deg/s
 Power/Wt Ratio 21.39 hp/t
 Pivot Yes
 Hull Armor // mm
 Turret Armor 280/90/75 mm
 Shells {{#ifeq:ARMOR_PIERCING_CR|ARMOR_PIERCING|| {{#ifeq:HIGH_EXPLOSIVE|ARMOR_PIERCING|| APCR/APCR/HE
 Shell Cost 1015/4800/870
 Damage 400/400/515 HP
 Penetration 248/297/60 mm
 Rate of Fire | |}} | |}} 4 r/m
 Damage Per Minute | |}} | |}} 1600
 Accuracy | |}} 0.44 m
 Aim time | |}} 2.3 s
 Turret Traverse 22 deg/s
 Gun Arc 360 °
 Elevation Arc -10 ° /+20 °
 Ammo Capacity 38 rounds
 Chance of Fire 12 %
 View Range 360 m
 Signal Range | |}} 850 m
Pros and Cons Pros: Good alpha damage, penetration, shell velocity, and aim time Excellent turret face - very well-armored with no weakspots Great gun depression (-10 degrees) Good hitpoint pool (1600 hitpoints stock) Cons: Poor DPM, accuracy, and on the move dispersion Sides and rear of both hull and turret are poorly armored Very prone to fires and engine damage Very vulnerable to SPGs Early Research None needed, it is a premium tank. Suggested Equipment</context>

<｜User｜>
{question}

<｜Assistant｜>
"""

prompt = PromptTemplate.from_template(template)

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="models\DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    max_tokens=10000,
    n_ctx = 4096,
    verbose=True,  # Verbose is required to pass to the callback manager
)

llm_chain = prompt | llm
# question = "What are the weaknesses of the Tornvagn?"
question = input()
llm_chain.invoke({"question": question})

