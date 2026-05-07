from abc import ABC, abstractmethod
from .utils import *
import requests
import base64
import json
#from lmdeploy.vl import load_image

import warnings
warnings.filterwarnings("ignore")

class Agent(ABC):
  def __init__(self, name, model_name):
    self.name = name
    self.model_name = model_name
    #self.config = kwargs
    self.model = None
    if name == "EvaluationAgent" or name == "VerificationAgent" or name == "ReasoningGenerationAgent" or name == "CompilationFixAgent":
      self.model = None
    else:
      self.model = self._load_model(model_name)
    #print(f"{name} loaded.")
    

  def _load_model(self, model_name):
      import os
      from openai import OpenAI, AzureOpenAI

      if model_name == "gpt-4o":
          # This open-source release supports both the standard OpenAI API and
          # Azure OpenAI. By default, it uses the standard OpenAI API:
          #   export OPENAI_API_KEY="<your-openai-api-key>"
          #   export OPENAI_MODEL="gpt-4o"
          #
          # To use Azure OpenAI instead, set AZURE_OPENAI_ENDPOINT:
          #   export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
          #   export AZURE_OPENAI_DEPLOYMENT="<your-gpt-4o-deployment-name>"
          #   export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
          #   export AZURE_OPENAI_API_KEY="<your-api-key>"
          #
          # If AZURE_OPENAI_API_KEY is not set, Azure mode falls back to Azure
          # Entra ID auth via Azure CLI or Managed Identity.
          endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("ENDPOINT_URL")
          if endpoint:
              deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("CRITIC_MODEL")
              if not deployment_name:
                  raise ValueError(
                      "Missing Azure OpenAI configuration: AZURE_OPENAI_DEPLOYMENT. "
                      "Please set it to your Azure deployment name for gpt-4o."
                  )

              client_kwargs = dict(
                  api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                  azure_endpoint=endpoint,
              )
              azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
              if azure_api_key:
                  client_kwargs["api_key"] = azure_api_key
              else:
                  from azure.identity import (
                      ChainedTokenCredential,
                      AzureCliCredential,
                      ManagedIdentityCredential,
                      get_bearer_token_provider,
                  )

                  credential = ChainedTokenCredential(
                      AzureCliCredential(),
                      ManagedIdentityCredential(),
                  )
                  token_provider = get_bearer_token_provider(
                      credential,
                      "https://cognitiveservices.azure.com/.default",
                  )
                  client_kwargs["azure_ad_token_provider"] = token_provider

              client = AzureOpenAI(**client_kwargs)
              return client, deployment_name
          else:
              api_key = os.getenv("OPENAI_API_KEY")
              if not api_key:
                  raise ValueError(
                      "Missing OpenAI API configuration: OPENAI_API_KEY. "
                      "Set OPENAI_API_KEY for the standard OpenAI API, or set "
                      "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT for Azure OpenAI."
                  )

              client = OpenAI(
                  api_key=api_key,
                  base_url=os.getenv("OPENAI_BASE_URL"),
              )
              openai_model = os.getenv("OPENAI_MODEL", model_name)
              return client, openai_model
      else:
          raise ValueError(f"Unsupported model: {model_name}")

  def t2t_generate(self, user_prompt, system_prompt="You are a LaTeX code expert."):
    model_name = self.model_name

    if model_name == "gpt-4o":
        client, deployment_name = self.model  # 解包 AzureOpenAI client 和 deployment 名称

        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                #{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    else:
      return None

  def critic_generate(self, user_prompt, system_prompt="You are a LaTeX table evaluation expert."):
    model_name = self.model_name

    if model_name == "gpt-4o":
        client, deployment_name = self.model  # 解包 AzureOpenAI client 和 deployment 名称
        
        import time

        for attempt in range(10):
            try:
                response = client.chat.completions.create(
                    model=deployment_name,
                    messages=[
                        #{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0,
                    timeout=60,  # 可调：设置每次请求的超时时间(秒)
                )
                return response.choices[0].message.content.strip()
            except:
                print(f"[Warning] Request timed out, retrying in 30 seconds...")
                time.sleep(30)
        return ""

  @abstractmethod
  def act(self, *args, **kwargs):
    pass
