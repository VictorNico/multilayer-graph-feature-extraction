#!/usr/bin/env python3
"""
=============================================================================
Module de Compilation LaTeX et Envoi d'Emails pour Pipeline MLNA
=============================================================================

Auteur: VICTOR DJIEMBOU (adaptation du script original)
Date de création: Non spécifiée
Dernière modification: Non spécifiée

Description:
    Ce module fournit des fonctionnalités pour compiler des documents LaTeX
    et les envoyer automatiquement par email via Gmail SMTP. Il est conçu
    pour automatiser l'envoi de rapports d'expérimentation ML.

    Fonctionnalités principales:
    - Compilation de fichiers LaTeX en PDF avec pdflatex
    - Gestion robuste des encodages pour les sorties LaTeX
    - Envoi d'emails avec pièces jointes (PDF et/ou TEX)
    - Support des destinataires en copie (CC)
    - Nettoyage automatique des fichiers temporaires LaTeX

Dépendances:
    - subprocess: Exécution de pdflatex en ligne de commande
    - smtplib: Protocole SMTP pour l'envoi d'emails
    - email: Construction de messages MIME multiparte
    - logging: Traçabilité des opérations
    - pathlib: Manipulation de chemins de fichiers

Configuration requise:
    - pdflatex doit être installé sur le système (TeX Live, MiKTeX, etc.)
    - Un compte Gmail avec un "mot de passe d'application" (App Password)
      car l'authentification 2FA standard n'est pas supportée par SMTP

Notes de sécurité:
    - Ne jamais hardcoder les identifiants Gmail dans le code
    - Utiliser des variables d'environnement ou des fichiers de config
    - Les mots de passe d'application Gmail sont requis (pas le mot de passe normal)

=============================================================================
"""

import os
import sys
import subprocess
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path

# Configuration du système de logging
# Format: timestamp - niveau - message
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LaTeXEmailSender:
    """
    Classe pour gérer la compilation LaTeX et l'envoi d'emails.

    Cette classe encapsule toutes les fonctionnalités nécessaires pour:
    1. Compiler des documents LaTeX en PDF
    2. Gérer les problèmes d'encodage courants
    3. Envoyer les résultats par email via Gmail

    Attributes:
        gmail_user (str): Adresse email Gmail de l'expéditeur
        gmail_password (str): Mot de passe d'application Gmail
        display_name (str): Nom d'affichage de l'expéditeur
    """

    def __init__(self, gmail_user, gmail_password, display_name="Pipeline MLNA"):
        """
        Initialise le gestionnaire LaTeX/Email.

        Args:
            gmail_user (str): Adresse Gmail complète (ex: user@gmail.com)
            gmail_password (str): Mot de passe d'application Gmail (16 caractères)
                                 PAS le mot de passe normal du compte!
            display_name (str, optional): Nom affiché dans le champ "De" de l'email
                                         (défaut: "Pipeline MLNA")

        Exemple:
            >>> sender = LaTeXEmailSender('ml_bot@gmail.com', 'xxxx xxxx xxxx xxxx',
            ...                           display_name='ML Experiment Bot')

        Note:
            Pour obtenir un mot de passe d'application Gmail:
            1. Activer la validation en 2 étapes sur votre compte
            2. Aller dans Sécurité > Mots de passe des applications
            3. Générer un nouveau mot de passe pour "Application personnalisée"
        """
        self.gmail_user = gmail_user
        self.gmail_password = gmail_password
        self.display_name = display_name

    def compile_latex(self, tex_file_path, output_dir=None, clean_temp=True):
        """
        Compile un fichier LaTeX en PDF avec gestion robuste des encodages.

        Cette méthode exécute pdflatex deux fois pour résoudre les références
        croisées, gère les problèmes d'encodage UTF-8, et nettoie les fichiers
        temporaires générés par LaTeX.

        Args:
            tex_file_path (str): Chemin absolu ou relatif vers le fichier .tex
            output_dir (str, optional): Répertoire de sortie pour le PDF
                                       Si None, utilise le même répertoire que le .tex
            clean_temp (bool, optional): Si True, supprime les fichiers .aux, .log, etc.
                                        (défaut: True)

        Returns:
            str: Chemin absolu vers le fichier PDF généré

        Raises:
            FileNotFoundError: Si le fichier .tex n'existe pas ou si le PDF n'est pas généré
            RuntimeError: Si la compilation pdflatex échoue
            UnicodeDecodeError: Si des problèmes d'encodage persistent

        Exemple:
            >>> sender = LaTeXEmailSender('user@gmail.com', 'password')
            >>> pdf = sender.compile_latex('/home/user/report.tex')
            >>> print(f"PDF créé: {pdf}")

        Processus de compilation:
            1. Vérification de l'existence du fichier .tex
            2. Première compilation pdflatex (génération du document)
            3. Deuxième compilation (résolution des références, TOC, etc.)
            4. Nettoyage des fichiers temporaires
            5. Vérification de la création du PDF

        Note:
            - pdflatex doit être accessible dans le PATH système
            - Le mode interaction=nonstopmode permet de compiler même avec des warnings
            - Les erreurs d'encodage sont gérées avec un fallback 'replace'
        """
        # Conversion du chemin en objet Path pour manipulation facile
        tex_path = Path(tex_file_path)

        # Vérification de l'existence du fichier source
        if not tex_path.exists():
            raise FileNotFoundError(f"Fichier LaTeX non trouvé: {tex_file_path}")

        # Détermination du répertoire de travail
        # Si output_dir n'est pas spécifié, on utilise le dossier du fichier .tex
        work_dir = tex_path.parent if output_dir is None else Path(output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)  # Création si nécessaire

        # Extraction du nom de fichier sans extension
        base_name = tex_path.stem

        # Construction du chemin du PDF de sortie
        pdf_path = work_dir / f"{base_name}.pdf"

        logger.info(f"Compilation de {tex_file_path}")

        try:
            # ============================================================
            # PREMIÈRE COMPILATION PDFLATEX
            # ============================================================
            logger.info("Première compilation LaTeX...")

            # Exécution de pdflatex avec options:
            # - interaction=nonstopmode: ne pas s'arrêter sur les erreurs mineures
            # - output-directory: où placer les fichiers générés
            result = subprocess.run([
                'pdflatex',
                '-interaction=nonstopmode',
                f'-output-directory={work_dir}',
                str(tex_path)
            ], capture_output=True, text=False, cwd=work_dir)  # text=False pour gérer nous-mêmes l'encodage

            # Gestion robuste du décodage de la sortie
            # Tentative en UTF-8, puis fallback avec remplacement des caractères invalides
            try:
                stdout_decoded = result.stdout.decode('utf-8')
                stderr_decoded = result.stderr.decode('utf-8')
            except UnicodeDecodeError:
                # En cas d'échec UTF-8, on remplace les caractères problématiques
                stdout_decoded = result.stdout.decode('utf-8', errors='replace')
                stderr_decoded = result.stderr.decode('utf-8', errors='replace')
                logger.warning("Caractères d'encodage remplacés dans la sortie pdflatex")

            # Vérification du code de retour (0 = succès)
            if result.returncode != 0:
                logger.error("Erreur lors de la première compilation:")
                logger.error("STDOUT:")
                logger.error(stdout_decoded)
                logger.error("STDERR:")
                logger.error(stderr_decoded)

                # Tentative de lecture du fichier .log pour plus de détails
                log_file = work_dir / f"{base_name}.log"
                if log_file.exists():
                    try:
                        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                            log_content = f.read()
                            # Affichage des 1000 derniers caractères (fin du log)
                            logger.error("Contenu du fichier .log (dernières 1000 caractères):")
                            logger.error(log_content[-1000:])
                    except Exception as log_e:
                        logger.warning(f"Impossible de lire le fichier .log: {log_e}")

                raise RuntimeError("Échec de la compilation LaTeX")

            logger.info("Première compilation réussie")

            # ============================================================
            # DEUXIÈME COMPILATION PDFLATEX (pour les références)
            # ============================================================
            # LaTeX nécessite souvent 2 compilations pour:
            # - Résoudre les références croisées (\ref, \cite)
            # - Générer la table des matières
            # - Mettre à jour les numéros de pages
            logger.info("Deuxième compilation pour les références...")

            result = subprocess.run([
                'pdflatex',
                '-interaction=nonstopmode',
                f'-output-directory={work_dir}',
                str(tex_path)
            ], capture_output=True, text=False, cwd=work_dir)

            # Décodage de la sortie de la deuxième compilation
            try:
                stdout_decoded = result.stdout.decode('utf-8')
                stderr_decoded = result.stderr.decode('utf-8')
            except UnicodeDecodeError:
                stdout_decoded = result.stdout.decode('utf-8', errors='replace')
                stderr_decoded = result.stderr.decode('utf-8', errors='replace')

            # La deuxième compilation peut échouer avec des warnings mais produire un PDF
            if result.returncode != 0:
                logger.warning("Avertissement lors de la deuxième compilation:")
                logger.warning("STDOUT:")
                logger.warning(stdout_decoded)
                logger.warning("STDERR:")
                logger.warning(stderr_decoded)
            else:
                logger.info("Deuxième compilation réussie")

            # ============================================================
            # NETTOYAGE DES FICHIERS TEMPORAIRES
            # ============================================================
            if clean_temp:
                self._clean_temp_files(work_dir, base_name)

            # Vérification finale de l'existence du PDF
            if not pdf_path.exists():
                raise FileNotFoundError(f"Le fichier PDF n'a pas été généré: {pdf_path}")

            logger.info(f"PDF généré avec succès: {pdf_path}")
            return str(pdf_path)

        except subprocess.CalledProcessError as e:
            logger.error(f"Erreur lors de l'exécution de pdflatex: {e}")
            raise
        except UnicodeDecodeError as e:
            logger.error(f"Erreur d'encodage lors de la lecture de la sortie pdflatex: {e}")
            logger.error("Le fichier LaTeX ou la sortie de pdflatex contient des caractères non-UTF-8")
            raise
        except Exception as e:
            logger.error(f"Erreur inattendue: {e}")
            raise

    def compile_latex_robust(self, tex_file_path, output_dir=None, clean_temp=True):
        """
        Version encore plus robuste de la compilation LaTeX avec fallbacks d'encodage.

        Cette méthode alternative essaie plusieurs stratégies d'encodage pour gérer
        les fichiers LaTeX avec des caractères spéciaux ou problématiques.

        Args:
            tex_file_path (str): Chemin vers le fichier .tex
            output_dir (str, optional): Répertoire de sortie
            clean_temp (bool, optional): Nettoyer les fichiers temporaires

        Returns:
            str: Chemin vers le PDF généré

        Stratégie d'encodage:
            Essaie successivement: UTF-8 → Latin-1 → CP1252 → ISO-8859-1 → Bytes bruts

        Note:
            Cette méthode est plus lente mais plus fiable pour les fichiers problématiques
        """
        # Conversion et vérification du chemin
        tex_path = Path(tex_file_path)

        if not tex_path.exists():
            raise FileNotFoundError(f"Fichier LaTeX non trouvé: {tex_file_path}")

        # Configuration des répertoires
        work_dir = tex_path.parent if output_dir is None else Path(output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        base_name = tex_path.stem
        pdf_path = work_dir / f"{base_name}.pdf"

        logger.info(f"Compilation de {tex_file_path}")

        def run_pdflatex_safe(attempt_num):
            """
            Fonction interne pour exécuter pdflatex avec gestion robuste de l'encodage.

            Args:
                attempt_num (int): Numéro de la tentative (pour logging)

            Returns:
                subprocess.CompletedProcess: Résultat de l'exécution pdflatex

            Stratégie:
                Teste plusieurs encodages courants pour décoder la sortie pdflatex.
                Si tous échouent, utilise le mode bytes brut avec remplacement des caractères.
            """
            logger.info(f"Tentative de compilation #{attempt_num}")

            # Liste d'encodages à essayer, du plus probable au moins probable
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

            # Tentative avec chaque encodage
            for encoding in encodings_to_try:
                try:
                    result = subprocess.run([
                        'pdflatex',
                        '-interaction=nonstopmode',
                        f'-output-directory={work_dir}',
                        str(tex_path)
                    ], capture_output=True, cwd=work_dir,
                        encoding=encoding, errors='replace')  # errors='replace' pour tolérer les erreurs

                    logger.info(f"Compilation réussie avec encodage: {encoding}")
                    return result

                except UnicodeDecodeError:
                    logger.warning(f"Encodage {encoding} échoué, essai suivant...")
                    continue
                except Exception as e:
                    logger.error(f"Erreur avec encodage {encoding}: {e}")
                    continue

            # Si tous les encodages ont échoué, utiliser le mode bytes brut
            logger.warning("Tous les encodages ont échoué, utilisation du mode bytes")
            result = subprocess.run([
                'pdflatex',
                '-interaction=nonstopmode',
                f'-output-directory={work_dir}',
                str(tex_path)
            ], capture_output=True, cwd=work_dir)

            # Conversion manuelle en string avec remplacement des caractères invalides
            result.stdout = result.stdout.decode('utf-8', errors='replace')
            result.stderr = result.stderr.decode('utf-8', errors='replace')

            return result

        try:
            # Première compilation avec gestion robuste
            result = run_pdflatex_safe(1)

            if result.returncode != 0:
                logger.error("Erreur lors de la première compilation:")
                logger.error("STDOUT:")
                logger.error(result.stdout)
                logger.error("STDERR:")
                logger.error(result.stderr)
                raise RuntimeError("Échec de la compilation LaTeX")

            logger.info("Première compilation réussie")

            # Deuxième compilation pour les références
            logger.info("Deuxième compilation pour les références...")
            result = run_pdflatex_safe(2)

            if result.returncode != 0:
                logger.warning("Avertissement lors de la deuxième compilation")
                logger.warning(result.stdout)
            else:
                logger.info("Deuxième compilation réussie")

            # Nettoyage des fichiers temporaires
            if clean_temp:
                self._clean_temp_files(work_dir, base_name)

            # Vérification de la création du PDF
            if not pdf_path.exists():
                raise FileNotFoundError(f"Le fichier PDF n'a pas été généré: {pdf_path}")

            logger.info(f"PDF généré avec succès: {pdf_path}")
            return str(pdf_path)

        except Exception as e:
            logger.error(f"Erreur lors de la compilation: {e}")
            raise

    def _clean_temp_files(self, work_dir, base_name):
        """
        Nettoie les fichiers temporaires générés par LaTeX.

        Args:
            work_dir (Path): Répertoire contenant les fichiers temporaires
            base_name (str): Nom de base du fichier (sans extension)

        Fichiers supprimés:
            .aux  - Fichiers auxiliaires (références)
            .log  - Logs de compilation
            .out  - Outline pour les PDFs
            .toc  - Table of Contents
            .nav  - Navigation (beamer)
            .snm  - Short navigation (beamer)
            .fls  - File list
            .fdb_latexmk - Base de données latexmk

        Note:
            Cette méthode ne lève pas d'exception si un fichier n'existe pas
        """
        # Liste des extensions de fichiers temporaires LaTeX
        temp_extensions = ['.aux', '.log', '.out', '.toc', '.nav', '.snm', '.fls', '.fdb_latexmk']

        for ext in temp_extensions:
            temp_file = work_dir / f"{base_name}{ext}"
            if temp_file.exists():
                temp_file.unlink()  # Suppression du fichier
                logger.debug(f"Supprimé: {temp_file}")

    def send_email(self, to_emails, subject, body, pdf_path=None, tex_path=None, cc_emails=None):
        """
        Envoie un email avec pièces jointes via Gmail SMTP.

        Construit un email MIME multiparte avec corps texte et pièces jointes
        (PDF et/ou TEX), puis l'envoie via le serveur SMTP de Gmail.

        Args:
            to_emails (list): Liste des adresses email des destinataires principaux
                             Ex: ['user1@example.com', 'user2@example.com']
            subject (str): Sujet de l'email
            body (str): Corps de l'email (texte brut)
            pdf_path (str, optional): Chemin vers le fichier PDF à joindre
            tex_path (str, optional): Chemin vers le fichier TEX à joindre
            cc_emails (list, optional): Liste des destinataires en copie (CC)

        Raises:
            smtplib.SMTPAuthenticationError: Si les identifiants Gmail sont invalides
            smtplib.SMTPException: En cas d'erreur SMTP générique
            FileNotFoundError: Si un fichier à joindre n'existe pas

        Exemple:
            >>> sender = LaTeXEmailSender('bot@gmail.com', 'password')
            >>> sender.send_email(
            ...     to_emails=['researcher@university.edu'],
            ...     subject='Résultats expérience ML',
            ...     body='Veuillez trouver ci-joint le rapport.',
            ...     pdf_path='/tmp/report.pdf',
            ...     cc_emails=['supervisor@university.edu']
            ... )

        Configuration Gmail requise:
            - Compte Gmail avec validation en 2 étapes activée
            - Mot de passe d'application généré (pas le mot de passe normal)
            - Serveur SMTP: smtp.gmail.com:587 (STARTTLS)

        Note:
            Les fichiers sont encodés en base64 pour le transport MIME
        """
        # Création du message MIME multiparte (permet pièces jointes)
        msg = MIMEMultipart()

        # Configuration de l'en-tête "From" avec nom d'affichage
        if self.display_name:
            msg['From'] = f"{self.display_name} <{self.gmail_user}>"
        else:
            msg['From'] = self.gmail_user

        # Configuration des destinataires principaux
        msg['To'] = ', '.join(to_emails)

        # Configuration des destinataires en copie (si fournis)
        if cc_emails:
            msg['Cc'] = ', '.join(cc_emails)

        # Sujet de l'email
        msg['Subject'] = subject

        # Ajout du corps de l'email en texte brut (UTF-8)
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        # ============================================================
        # AJOUT DE LA PIÈCE JOINTE PDF
        # ============================================================
        if pdf_path and os.path.exists(pdf_path):
            # Lecture du fichier PDF en mode binaire
            with open(pdf_path, "rb") as attachment:
                # Création d'une partie MIME de type binaire
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())

            # Encodage en base64 pour le transport par email
            encoders.encode_base64(part)

            # Extraction du nom de fichier pour l'en-tête Content-Disposition
            filename = os.path.basename(pdf_path)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename={filename}'
            )

            # Ajout de la pièce jointe au message
            msg.attach(part)
            logger.info(f"PDF joint: {filename}")

        # ============================================================
        # AJOUT DE LA PIÈCE JOINTE TEX
        # ============================================================
        if tex_path and os.path.exists(tex_path):
            # Même processus que pour le PDF
            with open(tex_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())

            encoders.encode_base64(part)
            filename = os.path.basename(tex_path)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename={filename}'
            )
            msg.attach(part)
            logger.info(f"Fichier TEX joint: {filename}")

        # ============================================================
        # ENVOI DE L'EMAIL VIA SMTP
        # ============================================================
        try:
            logger.info("Connexion au serveur SMTP Gmail...")

            # Connexion au serveur SMTP de Gmail (port 587 pour STARTTLS)
            server = smtplib.SMTP('smtp.gmail.com', 587)

            # Activation du chiffrement TLS (obligatoire pour Gmail)
            server.starttls()

            # Authentification avec identifiants Gmail
            server.login(self.gmail_user, self.gmail_password)

            # Liste complète des destinataires (To + CC)
            # Nécessaire pour que les serveurs SMTP délivrent à tous
            all_recipients = to_emails + (cc_emails or [])

            # Conversion du message en string pour envoi
            text = msg.as_string()

            # Envoi de l'email
            server.sendmail(self.gmail_user, all_recipients, text)

            # Fermeture de la connexion SMTP
            server.quit()

            logger.info(f"Email envoyé avec succès à: {', '.join(to_emails)}")

        except smtplib.SMTPAuthenticationError:
            # Identifiants invalides ou mot de passe d'application requis
            logger.error("Erreur d'authentification Gmail. Vérifiez vos identifiants.")
            logger.error("Note: Vous devez utiliser un 'mot de passe d'application', pas votre mot de passe normal.")
            raise
        except smtplib.SMTPException as e:
            # Erreur SMTP générique (réseau, serveur, etc.)
            logger.error(f"Erreur SMTP: {e}")
            raise
        except Exception as e:
            # Toute autre erreur inattendue
            logger.error(f"Erreur lors de l'envoi de l'email: {e}")
            raise


def SendReport(
        GMAIL_USER_,
        GMAIL_APP_PASSWORD_,
        LATEX_FILE_,
        TO_EMAILS_,
        CC_EMAILS_,
        SUBJECT_,
        EMAIL_BODY_
):
    """
    Fonction principale pour compiler un LaTeX et envoyer le rapport par email.

    Cette fonction orchestr tous les étapes du processus:
    1. Compilation du document LaTeX en PDF
    2. Préparation de l'email avec pièces jointes
    3. Envoi via Gmail SMTP

    Args:
        GMAIL_USER_ (str): Adresse Gmail de l'expéditeur
        GMAIL_APP_PASSWORD_ (str): Mot de passe d'application Gmail
        LATEX_FILE_ (str): Chemin vers le fichier .tex à compiler
        TO_EMAILS_ (list): Liste des destinataires principaux
        CC_EMAILS_ (list): Liste des destinataires en copie (peut être vide)
        SUBJECT_ (str): Sujet de l'email
        EMAIL_BODY_ (str): Corps du message

    Returns:
        None

    Raises:
        FileNotFoundError: Si le fichier LaTeX n'existe pas
        RuntimeError: Si la compilation LaTeX échoue
        smtplib.SMTPException: Si l'envoi de l'email échoue

    Exemple d'utilisation:
        >>> SendReport(
        ...     GMAIL_USER_='ml_bot@gmail.com',
        ...     GMAIL_APP_PASSWORD_='xxxx xxxx xxxx xxxx',
        ...     LATEX_FILE_='/project/reports/experiment_results.tex',
        ...     TO_EMAILS_=['researcher@university.edu', 'advisor@university.edu'],
        ...     CC_EMAILS_=['lab_head@university.edu'],
        ...     SUBJECT_='Résultats expérience ML - Run #42',
        ...     EMAIL_BODY_='Veuillez trouver ci-joint le rapport détaillé de l\'expérience.'
        ... )

    Note:
        En cas d'erreur, la fonction log les détails et termine avec sys.exit(1)
    """

    # ================================
    # CONFIGURATION
    # ================================

    # Assignation des paramètres d'entrée à des variables locales
    # (facilite la lisibilité et permet des modifications si nécessaire)
    GMAIL_USER = GMAIL_USER_
    GMAIL_APP_PASSWORD = GMAIL_APP_PASSWORD_  # Attention: mot de passe d'application requis!

    LATEX_FILE = LATEX_FILE_  # Fichier source LaTeX

    # Configuration des destinataires de l'email
    TO_EMAILS = TO_EMAILS_      # Destinataires principaux
    CC_EMAILS = CC_EMAILS_      # Destinataires en copie
    SUBJECT = SUBJECT_          # Sujet

    EMAIL_BODY = EMAIL_BODY_    # Corps du message


    # ================================
    # EXÉCUTION DU PIPELINE
    # ================================

    try:
        # Initialisation du gestionnaire LaTeX/Email
        sender = LaTeXEmailSender(GMAIL_USER, GMAIL_APP_PASSWORD)

        # ============================================================
        # ÉTAPE 1: COMPILATION LATEX
        # ============================================================
        logger.info("=== DÉBUT COMPILATION LATEX ===")
        pdf_path = sender.compile_latex(LATEX_FILE)
        logger.info("=== FIN COMPILATION LATEX ===")

        # ============================================================
        # ÉTAPE 2: ENVOI PAR EMAIL
        # ============================================================
        logger.info("=== DÉBUT ENVOI EMAIL ===")
        sender.send_email(
            to_emails=TO_EMAILS,
            cc_emails=CC_EMAILS,
            subject=SUBJECT,
            body=EMAIL_BODY,
            pdf_path=pdf_path,      # PDF compilé joint
            tex_path=LATEX_FILE     # Source LaTeX également joint
        )
        logger.info("=== FIN ENVOI EMAIL ===")

        # Message de succès
        logger.info("🎉 Processus terminé avec succès!")

    except FileNotFoundError as e:
        # Fichier LaTeX introuvable
        logger.error(f"❌ Fichier non trouvé: {e}")
        sys.exit(1)
    except RuntimeError as e:
        # Échec de la compilation LaTeX
        logger.error(f"❌ Erreur de compilation: {e}")
        sys.exit(1)
    except Exception as e:
        # Toute autre erreur inattendue
        logger.error(f"❌ Erreur inattendue: {e}")
        sys.exit(1)
