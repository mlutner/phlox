import React from 'react';
import { Box, Text, Button, VStack } from '@chakra-ui/react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    // Log the error details
    console.error('React Error Boundary caught an error:', error, errorInfo);
    this.setState({
      error: error,
      errorInfo: errorInfo
    });
  }

  render() {
    if (this.state.hasError) {
      return (
        <Box p={8} textAlign="center">
          <VStack spacing={4}>
            <Text fontSize="2xl" color="red.500">
              🚨 Something went wrong!
            </Text>
            <Text>The React app encountered an error:</Text>
            <Box bg="red.50" p={4} borderRadius="md" textAlign="left">
              <Text fontFamily="mono" fontSize="sm" color="red.700">
                {this.state.error && this.state.error.toString()}
              </Text>
              {this.state.errorInfo && (
                <details style={{ whiteSpace: 'pre-wrap', marginTop: '10px' }}>
                  <summary>Error Details</summary>
                  {this.state.errorInfo.componentStack}
                </details>
              )}
            </Box>
            <Button 
              colorScheme="blue" 
              onClick={() => window.location.reload()}
            >
              Reload App
            </Button>
          </VStack>
        </Box>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary; 