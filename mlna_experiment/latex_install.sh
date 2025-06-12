#!/bin/bash
set -e  # Arrête le script si une commande échoue

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonction pour afficher des messages colorés
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Fonction pour vérifier si une commande existe
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Fonction pour vérifier si un package est installé (Debian/Ubuntu)
package_installed() {
    dpkg -l "$1" >/dev/null 2>&1
}

# Fonction pour installer Homebrew sur macOS
install_homebrew() {
    print_info "Installation de Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Ajouter Homebrew au PATH pour la session actuelle
    if [[ -f "/opt/homebrew/bin/brew" ]]; then
        # Apple Silicon Mac
        export PATH="/opt/homebrew/bin:$PATH"
    elif [[ -f "/usr/local/bin/brew" ]]; then
        # Intel Mac
        export PATH="/usr/local/bin:$PATH"
    fi
}

# Fonction pour détecter le système d'exploitation
detect_os() {
    case "$(uname -s)" in
        Darwin*)
            echo "macos"
            ;;
        Linux*)
            if [[ -f /etc/os-release ]]; then
                . /etc/os-release
                echo $ID
            elif [[ -f /etc/debian_version ]]; then
                echo "debian"
            elif [[ -f /etc/redhat-release ]]; then
                echo "rhel"
            else
                echo "linux_unknown"
            fi
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Fonction pour installer LaTeX selon le système d'exploitation
install_latex() {
    local os=$(detect_os)

    case $os in
        macos)
            print_info "Installation de LaTeX sur macOS..."

            # Vérifier si Homebrew est installé
            if ! homebrew_installed; then
                print_warning "Homebrew n'est pas installé"
                read -p "Voulez-vous installer Homebrew? [y/N]: " -n 1 -r
                echo

                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    install_homebrew
                else
                    print_error "Homebrew est requis pour installer LaTeX via ce script"
                    print_info "Alternatives:"
                    print_info "1. Télécharger MacTeX depuis: https://www.tug.org/mactex/"
                    print_info "2. Installer Homebrew puis relancer ce script"
                    return 1
                fi
            fi

            # Proposer différentes options d'installation
            echo "Options d'installation LaTeX pour macOS:"
            echo "1) MacTeX (complet, ~4GB) - Recommandé"
            echo "2) BasicTeX (minimal, ~100MB) + packages essentiels"
            read -p "Choisissez une option [1-2]: " -n 1 -r
            echo

            case $REPLY in
                1)
                    print_info "Installation de MacTeX complet..."
                    brew install --cask mactex
                    ;;
                2)
                    print_info "Installation de BasicTeX + packages essentiels..."
                    brew install --cask basictex

                    # Ajouter BasicTeX au PATH pour cette session
                    export PATH="/usr/local/texlive/2023basic/bin/universal-darwin:$PATH"

                    # Installer les packages essentiels
                    print_info "Installation des packages LaTeX essentiels..."
                    sudo tlmgr update --self
                    sudo tlmgr install collection-latex collection-latexextra collection-fontsrecommended
                    ;;
                *)
                    print_error "Option invalide"
                    return 1
                    ;;
            esac

            # Mettre à jour le PATH
            print_info "Mise à jour du PATH..."
            if [[ -d "/usr/local/texlive" ]]; then
                export PATH="/usr/local/texlive/*/bin/*-darwin:$PATH"
            fi
            if [[ -d "/Library/TeX/texbin" ]]; then
                export PATH="/Library/TeX/texbin:$PATH"
            fi
            ;;
        ubuntu|debian)
            print_info "Installation de LaTeX sur $os..."
            sudo apt-get update
            sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended
            ;;
        fedora)
            print_info "Installation de LaTeX sur Fedora..."
            sudo dnf install -y texlive-latex texlive-collection-latexextra
            ;;
        centos|rhel)
            print_info "Installation de LaTeX sur RHEL/CentOS..."
            sudo yum install -y texlive-latex texlive-collection-latexextra
            ;;
        *)
            print_error "Système d'exploitation non supporté: $os"
            print_info "Systèmes supportés: macOS, Ubuntu, Debian, Fedora, CentOS, RHEL"
            print_info "Veuillez installer LaTeX manuellement pour votre système."
            return 1
            ;;
    esac
}

# Fonction principale de vérification LaTeX
check_latex_installation() {
    local os=$(detect_os)
    print_info "Vérification de l'installation LaTeX sur $os..."

    # Vérifier si pdflatex existe
    if command_exists pdflatex; then
        print_info "LaTeX est déjà installé ✓"
        pdflatex --version | head -n 1
        return 0
    fi

    # Vérifications spécifiques selon le système
    local latex_installed=false

    case $os in
        macos)
            if mactex_installed; then
                latex_installed=true
                print_warning "LaTeX semble installé mais pdflatex n'est pas dans le PATH"
                print_info "Tentative d'ajout des chemins LaTeX au PATH..."

                # Ajouter les chemins LaTeX courants au PATH
                if [[ -d "/Library/TeX/texbin" ]]; then
                    export PATH="/Library/TeX/texbin:$PATH"
                fi

                # Chercher les installations TeX Live
                for texlive_path in /usr/local/texlive/*/bin/*-darwin; do
                    if [[ -d "$texlive_path" ]]; then
                        export PATH="$texlive_path:$PATH"
                        break
                    fi
                done

                # Vérifier à nouveau
                if command_exists pdflatex; then
                    print_info "LaTeX est maintenant accessible ✓"
                    print_info "Ajoutez ceci à votre ~/.zshrc ou ~/.bash_profile:"
                    echo "export PATH=\"/Library/TeX/texbin:\$PATH\""
                    return 0
                fi
            fi
            ;;
        ubuntu|debian)
            if package_installed "texlive-latex-base"; then
                latex_installed=true
            fi
            ;;
        fedora|centos|rhel)
            if rpm -q texlive-latex >/dev/null 2>&1; then
                latex_installed=true
            fi
            ;;
    esac

    if $latex_installed; then
        print_warning "LaTeX semble installé mais pdflatex n'est pas accessible"
        print_info "Vous devrez peut-être redémarrer votre terminal ou recharger votre profil"
        return 0
    fi

    # LaTeX n'est pas installé
    print_warning "LaTeX n'est pas installé sur ce système"

    # Afficher des informations spécifiques au système
    case $os in
        macos)
            print_info "Options disponibles pour macOS:"
            print_info "1. Installation automatique via ce script (nécessite Homebrew)"
            print_info "2. Téléchargement manuel de MacTeX: https://www.tug.org/mactex/"
            ;;
        ubuntu|debian)
            print_info "Packages qui seront installés: texlive-latex-base texlive-latex-extra texlive-fonts-recommended"
            ;;
    esac

    # Demander à l'utilisateur s'il veut installer
    read -p "Voulez-vous installer LaTeX maintenant? [y/N]: " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_latex

        # Vérifier l'installation
        if command_exists pdflatex; then
            print_info "LaTeX installé avec succès ✓"
            pdflatex --version | head -n 1
        else
            print_error "L'installation de LaTeX a échoué ou pdflatex n'est pas dans le PATH"
            print_info "Redémarrez votre terminal et relancez le script"
            return 1
        fi
    else
        print_info "Installation annulée par l'utilisateur"
        return 1
    fi
}

# Fonction principale
main() {
    print_info "=== Script de vérification et configuration ==="

    # Vérifier LaTeX
    check_latex_installation

    print_info "Configuration terminée!"
}

# Exécution du script
main