import json
import asyncio
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime, timedelta

from pydantic import Field, BaseModel
from cachetools import TTLCache

from app.agent.react import ReActAgent
from app.logger import logger
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import AgentState, Message, ToolCall
from app.tool import CreateChatCompletion, Terminate, ToolCollection


TOOL_CALL_REQUIRED = "Tool calls required but none provided"


class ToolMetrics(BaseModel):
    """Métriques de performance pour un outil"""
    total_calls: int = 0
    total_errors: int = 0
    total_duration: float = 0.0
    average_duration: float = 0.0
    last_success: Optional[datetime] = None
    last_error: Optional[datetime] = None
    error_rate: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_efficiency: float = 0.0
    memory_usage: float = 0.0
    peak_memory_usage: float = 0.0
    concurrent_executions: int = 0
    max_concurrent_executions: int = 0
    response_times: List[float] = Field(default_factory=list)
    error_types: Dict[str, int] = Field(default_factory=dict)


class ToolCache(BaseModel):
    """Configuration du cache pour les outils"""
    enabled: bool = True
    ttl: int = 3600  # 1 heure par défaut
    max_size: int = 1000
    compression_enabled: bool = True
    compression_level: int = 6
    persistent_cache: bool = True
    persistent_path: str = "cache/tools"
    cleanup_interval: int = 3600  # Nettoyage toutes les heures
    max_memory_usage: float = 0.8  # 80% de la mémoire disponible


class ToolCallAgent(ReActAgent):
    """Agent optimisé pour l'exécution d'outils avec cache et métriques"""

    name: str = "toolcall"
    description: str = "Agent optimisé avec cache et métriques de performance"

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    # Configuration de base
    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate()
    )
    tool_choices: Literal["none", "auto", "required"] = "auto"
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])
    tool_calls: List[ToolCall] = Field(default_factory=list)
    max_steps: int = 30
    
    # Cache et métriques
    tool_cache_config: ToolCache = Field(default_factory=ToolCache)
    tool_metrics: Dict[str, ToolMetrics] = Field(default_factory=dict)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialisation du cache
        self.results_cache = TTLCache(
            maxsize=self.tool_cache_config.max_size,
            ttl=self.tool_cache_config.ttl
        )
        # Initialisation des métriques pour chaque outil
        for tool_name in self.available_tools.tool_map.keys():
            self.tool_metrics[tool_name] = ToolMetrics()

    async def think(self) -> bool:
        """Process current state and decide next actions using tools"""
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages += [user_msg]

        # Get response with tool options
        response = await self.llm.ask_tool(
            messages=self.messages,
            system_msgs=[Message.system_message(self.system_prompt)]
            if self.system_prompt
            else None,
            tools=self.available_tools.to_params(),
            tool_choice=self.tool_choices,
        )
        
        # Extraire le contenu et les tool_calls de la réponse
        content = response.content if hasattr(response, 'content') else None
        tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else []
        
        self.tool_calls = tool_calls

        # Log response info
        logger.info(f"✨ {self.name}'s thoughts: {content}")
        logger.info(
            f"🛠️ {self.name} selected {len(tool_calls) if tool_calls else 0} tools to use"
        )
        if tool_calls:
            logger.info(
                f"🧰 Tools being prepared: {[call.function.name for call in tool_calls]}"
            )

        try:
            # Handle different tool_choices modes
            if self.tool_choices == "none":
                if tool_calls:
                    logger.warning(
                        f"🤔 Hmm, {self.name} tried to use tools when they weren't available!"
                    )
                if content:
                    self.memory.add_message(Message.assistant_message(content))
                    return True
                return False

            # Create and add assistant message
            assistant_msg = (
                Message.from_tool_calls(
                    content=content, tool_calls=self.tool_calls
                )
                if self.tool_calls
                else Message.assistant_message(content)
            )
            self.memory.add_message(assistant_msg)

            if self.tool_choices == "required" and not self.tool_calls:
                return True  # Will be handled in act()

            # For 'auto' mode, continue with content if no commands but content exists
            if self.tool_choices == "auto" and not self.tool_calls:
                return bool(content)

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 Oops! The {self.name}'s thinking process hit a snag: {e}")
            self.memory.add_message(
                Message.assistant_message(
                    f"Error encountered while processing: {str(e)}"
                )
            )
            return False

    def _get_cache_key(self, name: str, args: Dict) -> str:
        """Génère une clé de cache unique pour un outil et ses arguments"""
        return f"{name}:{json.dumps(args, sort_keys=True)}"

    def _update_metrics(self, name: str, duration: float, success: bool, error: Optional[str] = None, cache_hit: bool = False, memory_usage: float = 0.0):
        """Met à jour les métriques de performance d'un outil"""
        import psutil
        
        metrics = self.tool_metrics[name]
        metrics.total_calls += 1
        metrics.total_duration += duration
        metrics.average_duration = metrics.total_duration / metrics.total_calls
        metrics.response_times.append(duration)
        
        # Limiter l'historique des temps de réponse
        if len(metrics.response_times) > 1000:
            metrics.response_times = metrics.response_times[-1000:]
        
        # Mise à jour des métriques de cache
        if cache_hit:
            metrics.cache_hits += 1
        else:
            metrics.cache_misses += 1
        metrics.cache_efficiency = metrics.cache_hits / metrics.total_calls if metrics.total_calls > 0 else 0.0
        
        # Mise à jour des métriques de mémoire
        metrics.memory_usage = memory_usage
        metrics.peak_memory_usage = max(metrics.peak_memory_usage, memory_usage)
        
        if success:
            metrics.last_success = datetime.now()
        else:
            metrics.total_errors += 1
            metrics.last_error = datetime.now()
            metrics.error_rate = metrics.total_errors / metrics.total_calls
            if error:
                metrics.error_types[error] = metrics.error_types.get(error, 0) + 1
        
        # Métriques de concurrence
        process = psutil.Process()
        metrics.concurrent_executions = len(process.threads())
        metrics.max_concurrent_executions = max(
            metrics.max_concurrent_executions,
            metrics.concurrent_executions
        )

    async def _cleanup_cache(self):
        """Nettoie le cache selon les règles définies"""
        try:
            import psutil
            
            # Vérifier l'utilisation de la mémoire
            memory = psutil.virtual_memory()
            if memory.percent >= self.tool_cache_config.max_memory_usage * 100:
                # Supprimer 20% des entrées les plus anciennes
                items_to_remove = int(len(self.results_cache) * 0.2)
                if items_to_remove > 0:
                    sorted_items = sorted(
                        self.results_cache.items(),
                        key=lambda x: x[1].get('last_used', 0)
                    )
                    for key, _ in sorted_items[:items_to_remove]:
                        self.results_cache.pop(key, None)
                    
            # Nettoyer les métriques anciennes
            current_time = datetime.now()
            for metrics in self.tool_metrics.values():
                metrics.response_times = [
                    t for t in metrics.response_times
                    if current_time - t < timedelta(days=7)
                ]
                
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage du cache: {e}")

    def _compress_result(self, result: str) -> bytes:
        """Compresse un résultat pour le stockage en cache"""
        import zlib
        return zlib.compress(
            result.encode(),
            level=self.tool_cache_config.compression_level
        )
        
    def _decompress_result(self, compressed: bytes) -> str:
        """Décompresse un résultat du cache"""
        import zlib
        return zlib.decompress(compressed).decode()

    async def execute_tool(self, command: ToolCall) -> str:
        """Execute un outil avec cache, métriques et gestion d'erreurs avancée"""
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format"

        name = command.function.name
        if name not in self.available_tools.tool_map:
            return f"Error: Unknown tool '{name}'"

        start_time = datetime.now()
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Parse des arguments
            args = json.loads(command.function.arguments or "{}")
            
            # Vérification du cache
            if self.tool_cache_config.enabled and name not in self.special_tool_names:
                cache_key = self._get_cache_key(name, args)
                cached_result = self.results_cache.get(cache_key)
                if cached_result:
                    logger.info(f"🎯 Cache hit pour l'outil '{name}'")
                    duration = (datetime.now() - start_time).total_seconds()
                    final_memory = process.memory_info().rss / 1024 / 1024
                    memory_usage = final_memory - initial_memory
                    self._update_metrics(
                        name, duration, True,
                        cache_hit=True,
                        memory_usage=memory_usage
                    )
                    if self.tool_cache_config.compression_enabled:
                        return self._decompress_result(cached_result)
                    return cached_result

            # Exécution de l'outil avec timeout
            async with asyncio.timeout(30):  # Timeout de 30 secondes
                result = await self.available_tools.execute(name=name, tool_input=args)

            # Formatage du résultat
            observation = (
                f"Observed output of cmd `{name}` executed:\n{str(result)}"
                if result
                else f"Cmd `{name}` completed with no output"
            )

            # Mise en cache du résultat avec compression
            if self.tool_cache_config.enabled and name not in self.special_tool_names:
                # Nettoyage périodique du cache
                if self.tool_metrics[name].total_calls % 100 == 0:  # Tous les 100 appels
                    await self._cleanup_cache()
                    
                if self.tool_cache_config.compression_enabled:
                    compressed = self._compress_result(observation)
                    self.results_cache[cache_key] = compressed
                else:
                    self.results_cache[cache_key] = observation

            # Gestion des outils spéciaux
            await self._handle_special_tool(name=name, result=result)
            
            # Mise à jour des métriques (succès)
            duration = (datetime.now() - start_time).total_seconds()
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_usage = final_memory - initial_memory
            self._update_metrics(
                name, duration, True,
                cache_hit=False,
                memory_usage=memory_usage
            )

            return observation

        except asyncio.TimeoutError:
            error_msg = f"Timeout lors de l'exécution de l'outil '{name}'"
            logger.error(error_msg)
            self._update_metrics(name, 30.0, False, "timeout")
            return f"Error: {error_msg}"
            
        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            logger.error(
                f"📝 Arguments invalides pour '{name}' - JSON invalide, arguments:{command.function.arguments}"
            )
            self._update_metrics(name, 0.0, False, "json_error")
            return f"Error: {error_msg}"
            
        except Exception as e:
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            logger.error(error_msg)
            duration = (datetime.now() - start_time).total_seconds()
            self._update_metrics(name, duration, False, str(e))
            return f"Error: {error_msg}"

    async def act(self) -> str:
        """Exécute les outils en parallèle quand possible"""
        if not self.tool_calls:
            if self.tool_choices == "required":
                raise ValueError(TOOL_CALL_REQUIRED)
            return self.messages[-1].content or "No content or commands to execute"

        # Groupe les outils qui peuvent être exécutés en parallèle
        parallel_tools = []
        sequential_tools = []
        
        for command in self.tool_calls:
            # Les outils spéciaux et ceux qui modifient l'état doivent être exécutés séquentiellement
            if self._is_special_tool(command.function.name) or command.function.name in ['write_file', 'delete_file']:
                sequential_tools.append(command)
            else:
                parallel_tools.append(command)

        results = []
        
        # Exécute les outils parallèles
        if parallel_tools:
            parallel_results = await asyncio.gather(
                *[self.execute_tool(cmd) for cmd in parallel_tools],
                return_exceptions=True
            )
            
            for cmd, result in zip(parallel_tools, parallel_results):
                if isinstance(result, Exception):
                    error_msg = f"Error in parallel execution of {cmd.function.name}: {str(result)}"
                    logger.error(error_msg)
                    results.append(error_msg)
                else:
                    results.append(result)
                    tool_msg = Message.tool_message(
                        content=result,
                        tool_call_id=cmd.id,
                        name=cmd.function.name
                    )
                    self.memory.add_message(tool_msg)

        # Exécute les outils séquentiels
        for command in sequential_tools:
            result = await self.execute_tool(command)
            results.append(result)
            tool_msg = Message.tool_message(
                content=result,
                tool_call_id=command.id,
                name=command.function.name
            )
            self.memory.add_message(tool_msg)

        return "\n\n".join(results)

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """Handle special tool execution and state changes"""
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # Set agent state to finished
            logger.info(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
        return True

    def _is_special_tool(self, name: str) -> bool:
        """Check if tool name is in special tools list"""
        return name.lower() in [n.lower() for n in self.special_tool_names]
