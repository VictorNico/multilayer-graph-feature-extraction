# ========================================
# Configuration des logs
# ========================================
LOG_DIR := logs
# 2 pour notre framework et 3 pour la recherche aléatoire
STEP := 2
TIMESTAMP := $(shell date '+%Y%m%d_%H%M%S')
USERNAME := $(shell whoami)

# Création du répertoire de logs
$(shell mkdir -p $(LOG_DIR))

# ========================================
# DATASETS AVEC LOGGING
# ========================================

# run our framework on adult
mlna_on_adu:
	@echo "🚀 Démarrage du framework sur Adult - $(shell date '+%Y-%m-%d %H:%M:%S')"
	@LOG_FILE="$(LOG_DIR)/Adult_$(TIMESTAMP).log" && \
	echo "📝 Log disponible dans: $$LOG_FILE" && \
	{ \
		echo "🔹 [$(shell date '+%Y-%m-%d %H:%M:%S')] DÉBUT du traitement Adult"; \
		echo "👤 Utilisateur: $(USERNAME)"; \
		echo "📁 Répertoire: $$(pwd)"; \
		echo ""; \
		 chmod +x launch.sh &&  ./launch.sh Adult $(STEP); \
		echo "✅ [$(shell date '+%Y-%m-%d %H:%M:%S')] FIN du traitement Adult"; \
	} > "$$LOG_FILE" 2>&1 && \
	echo "✅ Adult terminé - Log: $$LOG_FILE"

# stop our framework on Adult
stop-mlna_on_adu:
	@echo "🛑 Arrêt du framework sur Adult - $(shell date '+%Y-%m-%d %H:%M:%S')"
	@LOG_FILE="$(LOG_DIR)/Adult_stop_$(TIMESTAMP).log" && \
	{ \
		echo "🔹 [$(shell date '+%Y-%m-%d %H:%M:%S')] DÉBUT arrêt Adult"; \
		 chmod +x kill_screens.sh &&  ./kill_screens.sh Adult && rm -rf /Users/djiemboutienctheuvictornico/.screen/*; \
		echo "✅ [$(shell date '+%Y-%m-%d %H:%M:%S')] FIN arrêt Adult"; \
	} > "$$LOG_FILE" 2>&1

# run our framework on BankMarketing
mlna_on_ban:
	@echo "🚀 Démarrage du framework sur BankMarketing - $(shell date '+%Y-%m-%d %H:%M:%S')"
	@LOG_FILE="$(LOG_DIR)/BankMarketing_$(TIMESTAMP).log" && \
	echo "📝 Log disponible dans: $$LOG_FILE" && \
	{ \
		echo "🔹 [$(shell date '+%Y-%m-%d %H:%M:%S')] DÉBUT du traitement BankMarketing"; \
		echo "👤 Utilisateur: $(USERNAME)"; \
		echo "📁 Répertoire: $$(pwd)"; \
		echo ""; \
		 chmod +x launch.sh &&  ./launch.sh BankMarketing $(STEP); \
		echo "✅ [$(shell date '+%Y-%m-%d %H:%M:%S')] FIN du traitement BankMarketing"; \
	} > "$$LOG_FILE" 2>&1 && \
	echo "✅ BankMarketing terminé - Log: $$LOG_FILE"

# stop our framework on BankMarketing
stop-mlna_on_ban:
	@echo "🛑 Arrêt du framework sur BankMarketing"
	@LOG_FILE="$(LOG_DIR)/BankMarketing_stop_$(TIMESTAMP).log" && \
	{ \
		echo "🔹 [$(shell date '+%Y-%m-%d %H:%M:%S')] DÉBUT arrêt BankMarketing"; \
		 chmod +x kill_screens.sh &&  ./kill_screens.sh BankMarketing && rm -rf /Users/djiemboutienctheuvictornico/.screen/*; \
		echo "✅ [$(shell date '+%Y-%m-%d %H:%M:%S')] FIN arrêt BankMarketing"; \
	} > "$$LOG_FILE" 2>&1

# run our framework on CarEvaluation
mlna_on_car:
	@echo "🚀 Démarrage du framework sur CarEvaluation - $(shell date '+%Y-%m-%d %H:%M:%S')"
	@LOG_FILE="$(LOG_DIR)/CarEvaluation_$(TIMESTAMP).log" && \
	echo "📝 Log disponible dans: $$LOG_FILE" && \
	{ \
		echo "🔹 [$(shell date '+%Y-%m-%d %H:%M:%S')] DÉBUT du traitement CarEvaluation"; \
		echo "👤 Utilisateur: $(USERNAME)"; \
		echo "📁 Répertoire: $$(pwd)"; \
		echo ""; \
		 chmod +x launch.sh &&  ./launch.sh CarEvaluation $(STEP); \
		echo "✅ [$(shell date '+%Y-%m-%d %H:%M:%S')] FIN du traitement CarEvaluation"; \
	} > "$$LOG_FILE" 2>&1 && \
	echo "✅ CarEvaluation terminé - Log: $$LOG_FILE"

# stop our framework on CarEvaluation
stop-mlna_on_car:
	@echo "🛑 Arrêt du framework sur CarEvaluation"
	@LOG_FILE="$(LOG_DIR)/CarEvaluation_stop_$(TIMESTAMP).log" && \
	{ \
		echo "🔹 [$(shell date '+%Y-%m-%d %H:%M:%S')] DÉBUT arrêt CarEvaluation"; \
		 chmod +x kill_screens.sh &&  ./kill_screens.sh CarEvaluation && rm -rf /Users/djiemboutienctheuvictornico/.screen/*; \
		echo "✅ [$(shell date '+%Y-%m-%d %H:%M:%S')] FIN arrêt CarEvaluation"; \
	} > "$$LOG_FILE" 2>&1

# run our framework on CreditRiskDataset
mlna_on_crd:
	@echo "🚀 Démarrage du framework sur CreditRiskDataset - $(shell date '+%Y-%m-%d %H:%M:%S')"
	@LOG_FILE="$(LOG_DIR)/CreditRiskDataset_$(TIMESTAMP).log" && \
	echo "📝 Log disponible dans: $$LOG_FILE" && \
	{ \
		echo "🔹 [$(shell date '+%Y-%m-%d %H:%M:%S')] DÉBUT du traitement CreditRiskDataset"; \
		echo "👤 Utilisateur: $(USERNAME)"; \
		echo "📁 Répertoire: $$(pwd)"; \
		echo ""; \
		 chmod +x launch.sh &&  ./launch.sh CreditRiskDataset $(STEP); \
		echo "✅ [$(shell date '+%Y-%m-%d %H:%M:%S')] FIN du traitement CreditRiskDataset"; \
	} > "$$LOG_FILE" 2>&1 && \
	echo "✅ CreditRiskDataset terminé - Log: $$LOG_FILE"

# stop our framework on CreditRiskDataset
stop-mlna_on_crd:
	@echo "🛑 Arrêt du framework sur CreditRiskDataset"
	@LOG_FILE="$(LOG_DIR)/CreditRiskDataset_stop_$(TIMESTAMP).log" && \
	{ \
		echo "🔹 [$(shell date '+%Y-%m-%d %H:%M:%S')] DÉBUT arrêt CreditRiskDataset"; \
		 chmod +x kill_screens.sh &&  ./kill_screens.sh CreditRiskDataset && rm -rf /Users/djiemboutienctheuvictornico/.screen/*; \
		echo "✅ [$(shell date '+%Y-%m-%d %H:%M:%S')] FIN arrêt CreditRiskDataset"; \
	} > "$$LOG_FILE" 2>&1

# run our framework on Diabetes
mlna_on_dia:
	@echo "🚀 Démarrage du framework sur Diabetes - $(shell date '+%Y-%m-%d %H:%M:%S')"
	@LOG_FILE="$(LOG_DIR)/Diabetes_$(TIMESTAMP).log" && \
	echo "📝 Log disponible dans: $$LOG_FILE" && \
	{ \
		echo "🔹 [$(shell date '+%Y-%m-%d %H:%M:%S')] DÉBUT du traitement Diabetes"; \
		echo "👤 Utilisateur: $(USERNAME)"; \
		echo "📁 Répertoire: $$(pwd)"; \
		echo ""; \
		 chmod +x launch.sh &&  ./launch.sh Diabetes $(STEP); \
		echo "✅ [$(shell date '+%Y-%m-%d %H:%M:%S')] FIN du traitement Diabetes"; \
	} > "$$LOG_FILE" 2>&1 && \
	echo "✅ Diabetes terminé - Log: $$LOG_FILE"

# stop our framework on Diabetes
stop-mlna_on_dia:
	@echo "🛑 Arrêt du framework sur Diabetes"
	@LOG_FILE="$(LOG_DIR)/Diabetes_stop_$(TIMESTAMP).log" && \
	{ \
		echo "🔹 [$(shell date '+%Y-%m-%d %H:%M:%S')] DÉBUT arrêt Diabetes"; \
		 chmod +x kill_screens.sh &&  ./kill_screens.sh Diabetes && rm -rf /Users/djiemboutienctheuvictornico/.screen/*; \
		echo "✅ [$(shell date '+%Y-%m-%d %H:%M:%S')] FIN arrêt Diabetes"; \
	} > "$$LOG_FILE" 2>&1

# run our framework on GermanCredit
mlna_on_ger:
	@echo "🚀 Démarrage du framework sur GermanCredit - $(shell date '+%Y-%m-%d %H:%M:%S')"
	@LOG_FILE="$(LOG_DIR)/GermanCredit_$(TIMESTAMP).log" && \
	echo "📝 Log disponible dans: $$LOG_FILE" && \
	{ \
		echo "🔹 [$(shell date '+%Y-%m-%d %H:%M:%S')] DÉBUT du traitement GermanCredit"; \
		echo "👤 Utilisateur: $(USERNAME)"; \
		echo "📁 Répertoire: $$(pwd)"; \
		echo ""; \
		 chmod +x launch.sh &&  ./launch.sh GermanCredit $(STEP); \
		echo "✅ [$(shell date '+%Y-%m-%d %H:%M:%S')] FIN du traitement GermanCredit"; \
	} > "$$LOG_FILE" 2>&1 && \
	echo "✅ GermanCredit terminé - Log: $$LOG_FILE"

# stop our framework on GermanCredit
stop-mlna_on_ger:
	@echo "🛑 Arrêt du framework sur GermanCredit"
	@LOG_FILE="$(LOG_DIR)/GermanCredit_stop_$(TIMESTAMP).log" && \
	{ \
		echo "🔹 [$(shell date '+%Y-%m-%d %H:%M:%S')] DÉBUT arrêt GermanCredit"; \
		 chmod +x kill_screens.sh &&  ./kill_screens.sh GermanCredit && rm -rf /Users/djiemboutienctheuvictornico/.screen/*; \
		echo "✅ [$(shell date '+%Y-%m-%d %H:%M:%S')] FIN arrêt GermanCredit"; \
	} > "$$LOG_FILE" 2>&1

# run our framework on LoanDataforDummyBank
mlna_on_ld4:
	@echo "🚀 Démarrage du framework sur LoanDataforDummyBank - $(shell date '+%Y-%m-%d %H:%M:%S')"
	@LOG_FILE="$(LOG_DIR)/LoanDataforDummyBank_$(TIMESTAMP).log" && \
	echo "📝 Log disponible dans: $$LOG_FILE" && \
	{ \
		echo "🔹 [$(shell date '+%Y-%m-%d %H:%M:%S')] DÉBUT du traitement LoanDataforDummyBank"; \
		echo "👤 Utilisateur: $(USERNAME)"; \
		echo "📁 Répertoire: $$(pwd)"; \
		echo ""; \
		 chmod +x launch.sh &&  ./launch.sh LoanDataforDummyBank $(STEP); \
		echo "✅ [$(shell date '+%Y-%m-%d %H:%M:%S')] FIN du traitement LoanDataforDummyBank"; \
	} > "$$LOG_FILE" 2>&1 && \
	echo "✅ LoanDataforDummyBank terminé - Log: $$LOG_FILE"

# stop our framework on LoanDataforDummyBank
stop-mlna_on_ld4:
	@echo "🛑 Arrêt du framework sur LoanDataforDummyBank"
	@LOG_FILE="$(LOG_DIR)/LoanDataforDummyBank_stop_$(TIMESTAMP).log" && \
	{ \
		echo "🔹 [$(shell date '+%Y-%m-%d %H:%M:%S')] DÉBUT arrêt LoanDataforDummyBank"; \
		 chmod +x kill_screens.sh &&  ./kill_screens.sh LoanDataforDummyBank && rm -rf /Users/djiemboutienctheuvictornico/.screen/*; \
		echo "✅ [$(shell date '+%Y-%m-%d %H:%M:%S')] FIN arrêt LoanDataforDummyBank"; \
	} > "$$LOG_FILE" 2>&1

# run our framework on LoanDefaultDataset
mlna_on_ldd:
	@echo "🚀 Démarrage du framework sur LoanDefaultDataset - $(shell date '+%Y-%m-%d %H:%M:%S')"
	@LOG_FILE="$(LOG_DIR)/LoanDefaultDataset_$(TIMESTAMP).log" && \
	echo "📝 Log disponible dans: $$LOG_FILE" && \
	{ \
		echo "🔹 [$(shell date '+%Y-%m-%d %H:%M:%S')] DÉBUT du traitement LoanDefaultDataset"; \
		echo "👤 Utilisateur: $(USERNAME)"; \
		echo "📁 Répertoire: $$(pwd)"; \
		echo ""; \
		 chmod +x launch.sh &&  ./launch.sh LoanDefaultDataset $(STEP); \
		echo "✅ [$(shell date '+%Y-%m-%d %H:%M:%S')] FIN du traitement LoanDefaultDataset"; \
	} > "$$LOG_FILE" 2>&1 && \
	echo "✅ LoanDefaultDataset terminé - Log: $$LOG_FILE"

# stop our framework on LoanDefaultDataset
stop-mlna_on_ldd:
	@echo "🛑 Arrêt du framework sur LoanDefaultDataset"
	@LOG_FILE="$(LOG_DIR)/LoanDefaultDataset_stop_$(TIMESTAMP).log" && \
	{ \
		echo "🔹 [$(shell date '+%Y-%m-%d %H:%M:%S')] DÉBUT arrêt LoanDefaultDataset"; \
		 chmod +x kill_screens.sh &&  ./kill_screens.sh LoanDefaultDataset && rm -rf /Users/djiemboutienctheuvictornico/.screen/*; \
		echo "✅ [$(shell date '+%Y-%m-%d %H:%M:%S')] FIN arrêt LoanDefaultDataset"; \
	} > "$$LOG_FILE" 2>&1

# run our framework on Mushroom
mlna_on_mus:
	@echo "🚀 Démarrage du framework sur Mushroom - $(shell date '+%Y-%m-%d %H:%M:%S')"
	@LOG_FILE="$(LOG_DIR)/Mushroom_$(TIMESTAMP).log" && \
	echo "📝 Log disponible dans: $$LOG_FILE" && \
	{ \
		echo "🔹 [$(shell date '+%Y-%m-%d %H:%M:%S')] DÉBUT du traitement Mushroom"; \
		echo "👤 Utilisateur: $(USERNAME)"; \
		echo "📁 Répertoire: $$(pwd)"; \
		echo ""; \
		 chmod +x launch.sh &&  ./launch.sh Mushroom $(STEP); \
		echo "✅ [$(shell date '+%Y-%m-%d %H:%M:%S')] FIN du traitement Mushroom"; \
	} > "$$LOG_FILE" 2>&1 && \
	echo "✅ Mushroom terminé - Log: $$LOG_FILE"

# stop our framework on Mushroom
stop-mlna_on_mus:
	@echo "🛑 Arrêt du framework sur Mushroom"
	@LOG_FILE="$(LOG_DIR)/Mushroom_stop_$(TIMESTAMP).log" && \
	{ \
		echo "🔹 [$(shell date '+%Y-%m-%d %H:%M:%S')] DÉBUT arrêt Mushroom"; \
		 chmod +x kill_screens.sh &&  ./kill_screens.sh Mushroom && rm -rf /Users/djiemboutienctheuvictornico/.screen/*; \
		echo "✅ [$(shell date '+%Y-%m-%d %H:%M:%S')] FIN arrêt Mushroom"; \
	} > "$$LOG_FILE" 2>&1

# run our framework on Nursery
mlna_on_nur:
	@echo "🚀 Démarrage du framework sur Nursery - $(shell date '+%Y-%m-%d %H:%M:%S')"
	@LOG_FILE="$(LOG_DIR)/Nursery_$(TIMESTAMP).log" && \
	echo "📝 Log disponible dans: $$LOG_FILE" && \
	{ \
		echo "🔹 [$(shell date '+%Y-%m-%d %H:%M:%S')] DÉBUT du traitement Nursery"; \
		echo "👤 Utilisateur: $(USERNAME)"; \
		echo "📁 Répertoire: $$(pwd)"; \
		echo ""; \
		 chmod +x launch.sh &&  ./launch.sh Nursery $(STEP); \
		echo "✅ [$(shell date '+%Y-%m-%d %H:%M:%S')] FIN du traitement Nursery"; \
	} > "$$LOG_FILE" 2>&1 && \
	echo "✅ Nursery terminé - Log: $$LOG_FILE"

# stop our framework on Nursery
stop-mlna_on_nur:
	@echo "🛑 Arrêt du framework sur Nursery"
	@LOG_FILE="$(LOG_DIR)/Nursery_stop_$(TIMESTAMP).log" && \
	{ \
		echo "🔹 [$(shell date '+%Y-%m-%d %H:%M:%S')] DÉBUT arrêt Nursery"; \
		 chmod +x kill_screens.sh &&  ./kill_screens.sh Nursery && rm -rf /Users/djiemboutienctheuvictornico/.screen/*; \
		echo "✅ [$(shell date '+%Y-%m-%d %H:%M:%S')] FIN arrêt Nursery"; \
	} > "$$LOG_FILE" 2>&1

# run our framework on StudentPerformance
mlna_on_stu:
	@echo "🚀 Démarrage du framework sur StudentPerformance - $(shell date '+%Y-%m-%d %H:%M:%S')"
	@LOG_FILE="$(LOG_DIR)/StudentPerformance_$(TIMESTAMP).log" && \
	echo "📝 Log disponible dans: $$LOG_FILE" && \
	{ \
		echo "🔹 [$(shell date '+%Y-%m-%d %H:%M:%S')] DÉBUT du traitement StudentPerformance"; \
		echo "👤 Utilisateur: $(USERNAME)"; \
		echo "📁 Répertoire: $$(pwd)"; \
		echo ""; \
		 chmod +x launch.sh &&  ./launch.sh StudentPerformance $(STEP); \
		echo "✅ [$(shell date '+%Y-%m-%d %H:%M:%S')] FIN du traitement StudentPerformance"; \
	} > "$$LOG_FILE" 2>&1 && \
	echo "✅ StudentPerformance terminé - Log: $$LOG_FILE"

# stop our framework on StudentPerformance
stop-mlna_on_stu:
	@echo "🛑 Arrêt du framework sur StudentPerformance"
	@LOG_FILE="$(LOG_DIR)/StudentPerformance_stop_$(TIMESTAMP).log" && \
	{ \
		echo "🔹 [$(shell date '+%Y-%m-%d %H:%M:%S')] DÉBUT arrêt StudentPerformance"; \
		 chmod +x kill_screens.sh &&  ./kill_screens.sh StudentPerformance && rm -rf /Users/djiemboutienctheuvictornico/.screen/*; \
		echo "✅ [$(shell date '+%Y-%m-%d %H:%M:%S')] FIN arrêt StudentPerformance"; \
	} > "$$LOG_FILE" 2>&1

# ========================================
# COMMANDES PARALLÈLES AVEC LOGGING GLOBAL
# ========================================

# run parallel execution by default
# nice -n 10, permet de dire au système que nos execution ne sont pas prioritaires.
run-all:
	@echo "🚀 Démarrage de toutes les exécutions en parallèle - $(shell date '+%Y-%m-%d %H:%M:%S')"
	@LOG_FILE="$(LOG_DIR)/run-all_$(TIMESTAMP).log" && \
	echo "📝 Log global disponible dans: $$LOG_FILE" && \
	{ \
		echo "🔹 [$(shell date '+%Y-%m-%d %H:%M:%S')] DÉBUT exécution parallèle de tous les datasets"; \
		echo "👤 Utilisateur: $(USERNAME)"; \
		echo "📁 Répertoire: $$(pwd)"; \
		echo "📊 Datasets: Adult, BankMarketing, CarEvaluation, CreditRiskDataset, Diabetes, GermanCredit, LoanDataforDummyBank, LoanDefaultDataset, Mushroom, Nursery, StudentPerformance"; \
		echo ""; \
		nice -n 10 $(MAKE) -j 1 mlna_on_crd mlna_on_car mlna_on_nur mlna_on_adu mlna_on_dia mlna_on_ld4 mlna_on_ban mlna_on_ger mlna_on_stu mlna_on_mus mlna_on_ldd; \
		echo "✅ [$(shell date '+%Y-%m-%d %H:%M:%S')] FIN exécution parallèle"; \
	} > "$$LOG_FILE" 2>&1 && \
	echo "🎉 Toutes les exécutions terminées - Log global: $$LOG_FILE"

stop-all:
	@echo "🛑 Arrêt de toutes les exécutions - $(shell date '+%Y-%m-%d %H:%M:%S')"
	@LOG_FILE="$(LOG_DIR)/stop-all_$(TIMESTAMP).log" && \
	echo "📝 Log d'arrêt disponible dans: $$LOG_FILE" && \
	{ \
		echo "🔹 [$(shell date '+%Y-%m-%d %H:%M:%S')] DÉBUT arrêt de tous les processus"; \
		echo "👤 Utilisateur: $(USERNAME)"; \
		echo ""; \
		$(MAKE) -j 1 stop-mlna_on_crd stop-mlna_on_car stop-mlna_on_nur stop-mlna_on_adu stop-mlna_on_dia stop-mlna_on_ld4 stop-mlna_on_ban stop-mlna_on_ger stop-mlna_on_stu stop-mlna_on_mus stop-mlna_on_ldd; \
		echo "✅ [$(shell date '+%Y-%m-%d %H:%M:%S')] FIN arrêt de tous les processus"; \
	} > "$$LOG_FILE" 2>&1 && \
	echo "🎉 Tous les processus arrêtés - Log: $$LOG_FILE"

# ========================================
# UTILITAIRES DE GESTION DES LOGS
# ========================================

# Voir les logs récents
show-logs:
	@echo "📋 Logs disponibles dans $(LOG_DIR):"
	@ls -lart $(LOG_DIR)/ | tail -10

# Nettoyer les anciens logs (plus de 7 jours)
clean-old-logs:
	@echo "🧹 Nettoyage des logs anciens (>7 jours)..."
	@find $(LOG_DIR) -name "*.log" -mtime +7 -delete
	@echo "✅ Nettoyage terminé"

# Voir le log le plus récent
tail-latest:
	@echo "👀 Suivi du log le plus récent:"
	@tail -f $$(ls -t $(LOG_DIR)/*.log | head -1)

# Résumé des exécutions du jour
daily-summary:
	@echo "📊 Résumé des exécutions du jour ($(shell date '+%Y%m%d')):"
	@find $(LOG_DIR) -name "*$(shell date '+%Y%m%d')*.log" -exec basename {} \; | sort

# Aide
help:
	@echo "🔧 Commandes disponibles:"
	@echo "  Exécution individuelle: mlna_on_<dataset>"
	@echo "  Arrêt individuel: stop-mlna_on_<dataset>"
	@echo "  Exécution parallèle: run-all"
	@echo "  Arrêt global: stop-all"
	@echo ""
	@echo "📝 Gestion des logs:"
	@echo "  show-logs: Afficher les logs récents"
	@echo "  tail-latest: Suivre le log le plus récent"
	@echo "  clean-old-logs: Nettoyer les logs anciens"
	@echo "  daily-summary: Résumé du jour"
	@echo ""
	@echo "📁 Les logs sont sauvegardés dans: $(LOG_DIR)/"

.PHONY: mlna_on_adu stop-mlna_on_adu mlna_on_ban stop-mlna_on_ban mlna_on_car stop-mlna_on_car mlna_on_crd stop-mlna_on_crd mlna_on_dia stop-mlna_on_dia mlna_on_ger stop-mlna_on_ger mlna_on_ld4 stop-mlna_on_ld4 mlna_on_ldd stop-mlna_on_ldd mlna_on_mus stop-mlna_on_mus mlna_on_nur stop-mlna_on_nur mlna_on_stu stop-mlna_on_stu run-all stop-all show-logs clean-old-logs tail-latest daily-summary help