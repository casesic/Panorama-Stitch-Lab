echo $SSH_CLIENT
cat > ~/.claude/settings.json << 'EOF'
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://172.28.208.101:15721",
    "ANTHROPIC_AUTH_TOKEN": "any-string"
  }
}
EOF
unset ANTHROPIC_API_KEY

netsh interface portproxy delete v4tov4 listenaddress=172.28.208.78 listenport=15721
netsh interface portproxy add v4tov4 listenaddress=172.28.208.101 connectaddress=127.0.0.1 connectport=15721 listenport=15721