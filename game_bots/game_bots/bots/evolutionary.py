from deap import creator, base, tools, cma
from deap.algorithms import eaGenerateUpdate
import numpy as np
from game_bots.bots.machine_learning import TDBotsmall


def evaluate(individuals):
    bot = TDBotsmall()
    bot.weights = np.array(individuals).reshape(bot.num_feat, bot.size_feat)
    game = bot.episode(False)
    return (game.score,)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
strategy = cma.Strategy(
    centroid=list(np.random.uniform(-5.0, 5.0, 6144)),
    sigma=np.random.uniform(0.0, 5.0, None),
    lambda_=6,
)
toolbox.register("generate", strategy.generate, creator.Individual)
toolbox.register("update", strategy.update)
toolbox.register("evaluate", lambda ind: evaluate(ind))


hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

eaGenerateUpdate(toolbox, 10, stats=stats, halloffame=hof)
