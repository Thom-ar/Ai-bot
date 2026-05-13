import discord
from discord.ext import commands
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D as KDepthwiseConv2D
from tensorflow.keras.utils import register_keras_serializable
from PIL import Image
import numpy as np

intents = discord.Intents.default()
intents.message_content = True

@register_keras_serializable(package='custom', name='DepthwiseConv2D')
class DepthwiseConv2DFix(KDepthwiseConv2D):
    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config.pop('groups', None)
        return super().from_config(config)


bot = commands.Bot(command_prefix='$', intents=intents)

def get_class(model_path, labels_path, image_path):
    # Carregar o modelo
    model = load_model(model_path, custom_objects={'DepthwiseConv2D': DepthwiseConv2DFix}, compile=False)
    
    # Carregar os labels
    with open(labels_path, 'r') as f:
        labels = [line.split(' ', 1)[1] if ' ' in line else line.strip() for line in f]
    
    # Abrir e pré-processar a imagem
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))  # Assumindo tamanho comum; ajuste se necessário
    image_array = np.array(image) / 255.0  # Normalizar
    image_array = np.expand_dims(image_array, axis=0)
    
    # Fazer predição
    predictions = model.predict(image_array)
    class_idx = np.argmax(predictions)
    
    return labels[class_idx]

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command()
async def hello(ctx):
    await ctx.send(f'Olá {bot.user}!')

@bot.command()
async def heh(ctx, count_heh = 5):
    await ctx.send("he" * count_heh)

@bot.command()
async def check(ctx):
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            file_name = attachment.filename
            file_url = attachment.url
            await attachment.save(f"./{attachment.filename}")
            await ctx.send(get_class(model_path="./keras_model.h5", labels_path="labels.txt", image_path=f"./{attachment.filename}"))
    else:
        await ctx.send("Você esqueceu de enviar a imagem :(")
bot.run("")
